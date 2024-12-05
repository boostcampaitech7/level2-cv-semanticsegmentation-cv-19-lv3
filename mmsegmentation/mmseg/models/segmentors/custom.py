from mmseg.registry import MODELS
from mmseg.models.segmentors import EncoderDecoder
from mmseg.structures.seg_data_sample import SegDataSample
from mmseg.models.utils.wrappers import resize

from mmengine.structures import PixelData
import torch



class PostProcessResultMixin:
    def postprocess_result(self,
                           seg_logits,
                           data_samples):
        """ Convert results list to `SegDataSample`.
        Args:
            seg_logits (Tensor): The segmentation results, seg_logits from
                model of each input image.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`. Default to None.
        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        batch_size, C, H, W = seg_logits.shape

        if data_samples is None:
            data_samples = [SegDataSample() for _ in range(batch_size)]
            only_prediction = True
        else:
            only_prediction = False

        for i in range(batch_size):
            if not only_prediction:
                img_meta = data_samples[i].metainfo
                # remove padding area
                if 'img_padding_size' not in img_meta:
                    padding_size = img_meta.get('padding_size', [0] * 4)
                else:
                    padding_size = img_meta['img_padding_size']
                padding_left, padding_right, padding_top, padding_bottom =\
                    padding_size
                # i_seg_logits shape is 1, C, H, W after remove padding
                i_seg_logits = seg_logits[i:i + 1, :,
                                          padding_top:H - padding_bottom,
                                          padding_left:W - padding_right]

                flip = img_meta.get('flip', None)
                if flip:
                    flip_direction = img_meta.get('flip_direction', None)
                    assert flip_direction in ['horizontal', 'vertical']
                    if flip_direction == 'horizontal':
                        i_seg_logits = i_seg_logits.flip(dims=(3, ))
                    else:
                        i_seg_logits = i_seg_logits.flip(dims=(2, ))

                # resize as original shape
                i_seg_logits = resize(
                    i_seg_logits,
                    size=img_meta['ori_shape'],
                    mode='bicubic',
                    align_corners=self.align_corners,
                    warning=False).squeeze(0)
            else:
                i_seg_logits = seg_logits[i]

            # if self.module.decode_head.threshold:
            #     threshold = self.module.decode_head.threshold
            # else:
            #     threshold = 0.5
            threshold = 0.45
            
            i_seg_logits = i_seg_logits.sigmoid()
            i_seg_pred = (i_seg_logits > threshold).to(i_seg_logits)
            
            data_samples[i].set_data({
                'seg_logits':
                PixelData(**{'data': i_seg_logits}),
                'pred_sem_seg':
                PixelData(**{'data': i_seg_pred})
            })

        return data_samples

# class PostProcessResultMixin:
#     def postprocess_result(self, seg_logits, data_samples):
#         """Convert results list to `SegDataSample`.
#         Args:
#             seg_logits (Tensor): The segmentation results, seg_logits from
#                 model of each input image.
#             data_samples (list[:obj:`SegDataSample`]): The seg data samples.
#                 It usually includes information such as `metainfo` and
#                 `gt_sem_seg`. Default to None.
#         Returns:
#             list[:obj:`SegDataSample`]: Segmentation results of the
#             input images. Each SegDataSample usually contain:

#             - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
#             - ``seg_logits``(PixelData): Predicted logits of semantic
#                 segmentation before normalization.
#         """
#         batch_size, C, H, W = seg_logits.shape

#         if data_samples is None:
#             data_samples = [SegDataSample() for _ in range(batch_size)]
#             only_prediction = True
#         else:
#             only_prediction = False

#         for i in range(batch_size):
#             if not only_prediction:
#                 img_meta = data_samples[i].metainfo
#                 # remove padding area
#                 if "img_padding_size" not in img_meta:
#                     padding_size = img_meta.get("padding_size", [0] * 4)
#                 else:
#                     padding_size = img_meta["img_padding_size"]
#                 padding_left, padding_right, padding_top, padding_bottom = padding_size
#                 # i_seg_logits shape is 1, C, H, W after remove padding
#                 i_seg_logits = seg_logits[
#                     i : i + 1,
#                     :,
#                     padding_top : H - padding_bottom,
#                     padding_left : W - padding_right,
#                 ]

#                 flip = img_meta.get("flip", None)
#                 if flip:
#                     flip_direction = img_meta.get("flip_direction", None)
#                     assert flip_direction in ["horizontal", "vertical"]
#                     if flip_direction == "horizontal":
#                         i_seg_logits = i_seg_logits.flip(dims=(3,))
#                     else:
#                         i_seg_logits = i_seg_logits.flip(dims=(2,))

#                 # resize as original shape
#                 i_seg_logits = resize(
#                     i_seg_logits,
#                     size=img_meta["ori_shape"],
#                     mode="bilinear",
#                     align_corners=self.align_corners,
#                     warning=False,
#                 ).squeeze(0)
#             else:
#                 i_seg_logits = seg_logits[i]

#             # i_seg_logits = i_seg_logits.sigmoid()
#             # i_seg_pred = (i_seg_logits > self.decode_head.threshold).to(i_seg_logits)
#             # Sigmoid로 변환하여 confidence 계산
#             i_seg_logits = i_seg_logits.sigmoid()  # [C, H, W]

#             # 각 픽셀별로 상위 2개의 클래스의 confidence 값과 인덱스 추출
#             topk_values, topk_indices = i_seg_logits.topk(2, dim=0)  # topk_values, topk_indices: [2, H, W]

#             # Threshold 기준을 적용하여 confidence 값이 threshold 이상인 경우만 남김
#             threshold_mask = topk_values > self.decode_head.threshold  # [2, H, W]

#             # i_seg_pred 초기화
#             i_seg_pred = torch.zeros_like(i_seg_logits)  # [C, H, W]

#             # threshold를 만족하는 상위 2개 클래스만 유지하고, 나머지는 0으로 처리
#             for j in range(2):  # 최대 2개의 클래스만 선택
#                 # topk_indices는 [2, H, W]에서 i번째 인덱스 추출, 이를 [1, H, W]로 유지
#                 class_indices = topk_indices[j:j+1, :, :]  # [1, H, W]
#                 class_mask = threshold_mask[j:j+1, :, :].to(i_seg_logits.dtype)  # [1, H, W]
                
#                 # 선택된 클래스 위치에 값을 설정
#                 i_seg_pred.scatter_(0, class_indices, class_mask)
                
#             data_samples[i].set_data(
#                 {
#                     "seg_logits": PixelData(**{"data": i_seg_logits}),
#                     "pred_sem_seg": PixelData(**{"data": i_seg_pred}),
#                 }
#             )

#         return data_samples


@MODELS.register_module()
class EncoderDecoderWithoutArgmax(PostProcessResultMixin, EncoderDecoder):
    pass