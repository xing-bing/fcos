from .resnet import resnet

__all__ = ["build_backbone"]
support_backbone = ['resnet', 'hrnet']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f"all support backbone is {support_backbone}"
    backbone = eval(backbone_name)(**kwargs)
    return backbone
