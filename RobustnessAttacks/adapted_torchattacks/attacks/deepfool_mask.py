import torch
import torch.nn as nn

from ..attack import Attack
from model_templates.utils_models import InverseNormalization, index


class DeepFool(Attack):
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (Default: 50)
        overshoot (float): parameter for enhancing the noise. (Default: 0.02)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        attack = torchattacks.DeepFool(model, steps=50, overshoot=0.02)
        adv_images = attack(images, labels)
    """
    def __init__(self, model, steps=2, overshoot=0.02, descision_threshold=0.3):
        super().__init__("DeepFool", model)
        self.steps = steps
        self.overshoot = overshoot
        self._supported_mode = ['default']
        self.inverse_normalization = InverseNormalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.descision_threshold = descision_threshold

    def forward(self, images, labels, return_target_labels=False):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        target_labels = labels.clone().detach().to(self.device)

        batch_size = len(images)
        correct = torch.tensor([True]*batch_size)
        curr_steps = 0

        adv_images = []
        for idx in range(batch_size):
            image = images[idx:idx+1].clone().detach()
            adv_images.append(image)

        while (True in correct) and (curr_steps < self.steps):
            for idx in range(batch_size):
                if not correct[idx]: continue
                early_stop, pre, adv_image = self._forward_indiv(adv_images[idx], labels[idx])
                adv_images[idx] = adv_image
                target_labels[idx] = pre
                if early_stop:
                    correct[idx] = False
            curr_steps += 1

        adv_images = torch.cat(adv_images).detach()

        if return_target_labels:
            return adv_images, target_labels

        return adv_images

    def _forward_indiv(self, image, label):
        self.model.eval()
        outputs, transformed_images, _ = self.model(image)  # eval mode
        outputs = outputs[0]  # only 1 element in batch
        transformed_images = transformed_images.tensors  # required to skip the normalization in the forward pass

        threshold_idx = len([x for x in outputs["scores"] if x > self.descision_threshold])
        thresholded_pred = outputs["labels"][:threshold_idx]

        if label.item() not in thresholded_pred:
            if len(outputs["labels"]) > 0:
                return True, outputs["labels"][0], image  # return highest score if no bird is found
            else:
                return True, 1, image # return 1 if there is no object detected

        ws = self._construct_jacobian(outputs["scores"][:threshold_idx], transformed_images)
        image = self.inverse_normalization(transformed_images).detach()

        f_0 = outputs["scores"][index(thresholded_pred, label.item())] # bird with highest score
        w_0 = ws[index(thresholded_pred, label.item())]

        wrong_classes = [i for i in range(len(thresholded_pred)) if i != label.item()]
        f_k = outputs["scores"][:threshold_idx][wrong_classes] # choose classes with scores above threshold
        w_k = ws[wrong_classes]

        f_prime = f_k - f_0
        w_prime = w_k - w_0
        value = torch.abs(f_prime) / torch.norm(nn.Flatten()(w_prime), p=2, dim=1)
        _, hat_L = torch.min(value, 0)

        delta = (torch.abs(f_prime[hat_L])*w_prime[hat_L] \
                 / (torch.norm(w_prime[hat_L], p=2)**2))

        target_label = hat_L if hat_L < label.item() else hat_L+1

        adv_image = image + (1+self.overshoot)*delta
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()
        return False, target_label, adv_image

    # https://stackoverflow.com/questions/63096122/pytorch-is-it-possible-to-differentiate-a-matrix
    # torch.autograd.functional.jacobian is only for torch >= 1.5.1
    def _construct_jacobian(self, y, x):
        x_grads = []
        for idx, y_element in enumerate(y):
            if x.grad is not None:
                x.grad.zero_()
            y_element.backward(retain_graph=(False or idx+1 < len(y)))
            x_grads.append(x.grad.clone().detach())
        return torch.stack(x_grads).reshape(*y.shape, *x.shape)