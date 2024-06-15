import os
import argparse
import torch as th
import torch.nn.functional as F
import time
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util
from a4_imgStanderize import ISODATA,standerize

# Workaround
try:
    import ctypes
    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass


from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402

def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):

    print("Start", conf['name'])

    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)

    for batch in iter(dl):

        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )


        result = sample_fn(
            model_fn,
            (batch_size, 3, conf.image_size, conf.image_size),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )
        srs = toU8(result['sample'])
        gts = toU8(result['gt'])
        lrs = toU8(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                   th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))

        gt_keep_masks = toU8((model_kwargs.get('gt_keep_mask') * 2 - 1))

        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False)

    print("sampling complete")


if __name__ == "__main__":
    # 将./a4_reference_Img文件夹中被Lama模型生成的基准图像与lt掩码结合，进行第一次重采样√
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--conf_path', type=str, required=False, default='./a4_ymls/1_lt.yml')
    # args = vars(parser.parse_args())
    #
    # conf_arg = conf_mgt.conf_base.Default_Conf()
    # conf_arg.update(yamlread(args.get('conf_path')))
    # main(conf_arg)

    # 现在，将./a4_1_lt_result文件夹的图片与rt掩码结合，进行第二次重采样,图片保存至/a4_2_rt_result√
    # parser2 = argparse.ArgumentParser()
    # parser2.add_argument('--conf_path', type=str, required=False, default='./a4_ymls/2_rt.yml')
    # args = vars(parser2.parse_args())
    # conf_arg = conf_mgt.conf_base.Default_Conf()
    # conf_arg.update(yamlread(args.get('conf_path')))
    # main(conf_arg)
    # print("--------第二次重采样完成--------")

    # 将第二次重采样后的像素，颜色压缩至40种
    # standerize = standerize('./a4_2_rt_result/000000.png', './a2_Target_Standerize_Mask/rt.png', 64,
    #                         './a4_2_rt_result/000000.png')
    # standerize.standerize()
    # print("--------二次采样后颜色压缩完成--------")

    # 将./a4_2_rt_result文件夹的图片与lb掩码结合，进行第三次重采样,图片保存至/a4_3_lb_result√
    # parser3 = argparse.ArgumentParser()
    # parser3.add_argument('--conf_path', type=str, required=False, default='./a4_ymls/3_lb.yml')
    # args = vars(parser3.parse_args())
    # conf_arg = conf_mgt.conf_base.Default_Conf()
    # conf_arg.update(yamlread(args.get('conf_path')))
    # main(conf_arg)
    # print("--------三次重采样完成--------")

    # 将第三次重采样后的像素，颜色压缩至30种
    # standerize = standerize('./a4_3_lb_result/000000.png', './a2_Target_Standerize_Mask/lb.png', 38,
    #                         './a4_3_lb_result/000000.png')
    # standerize.standerize()
    # print("--------三次采样后颜色压缩完成--------")

    # 进行第四次重采样,将./gt3_lb_result文件夹的图片与rb掩码结合，图片保存至/.gt4_rb_result
    # parser4 = argparse.ArgumentParser()
    # parser4.add_argument('--conf_path', type=str, required=False, default='./a4_ymls/4_rb.yml')
    # args = vars(parser4.parse_args())
    # conf_arg = conf_mgt.conf_base.Default_Conf()
    # conf_arg.update(yamlread(args.get('conf_path')))
    # main(conf_arg)
    # print("--------四次重采样完成--------")

    # 将第四次重采样后的像素，颜色压缩至16种
    standerize = standerize('./a4_4_rb_result/000000.png', './a2_Target_Standerize_Mask/rb.png', 25,
                            './a4_4_rb_result/000000.png')
    standerize.standerize()
    print("--------四次采样后颜色压缩完成--------")

