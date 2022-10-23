from src.conv_onet import models


def get_model(cfg, ilabel_plus=True, nice=False):
    """
    Return the network model.

    Args:
        cfg (dict): imported yaml config.
        nice (bool, optional): whether or not use Neural Implicit Scalable Encoding. Defaults to False.

    Returns:
        decoder (nn.module): the network model.
    """

    dim = cfg['data']['dim']
    coarse_grid_len = cfg['grid_len']['coarse']
    middle_grid_len = cfg['grid_len']['middle']
    fine_grid_len = cfg['grid_len']['fine']
    color_grid_len = cfg['grid_len']['color']
    seg_grid_len = cfg['grid_len']['seg']
    c_dim = cfg['model']['c_dim']  # feature dimensions # 32
    pos_embedding_method = cfg['model']['pos_embedding_method']

    if ilabel_plus:
        decoder = models.decoder_dict['ilabel_plus'](
            dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
            middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
            color_grid_len=color_grid_len, seg_grid_len=seg_grid_len, pos_embedding_method=pos_embedding_method
        )
    elif nice:
        decoder = models.decoder_dict['nice'](
            dim=dim, c_dim=c_dim, coarse=cfg['coarse'], coarse_grid_len=coarse_grid_len,
            middle_grid_len=middle_grid_len, fine_grid_len=fine_grid_len,
            color_grid_len=color_grid_len, pos_embedding_method=pos_embedding_method)
    else:
        #TODO: 여기가 아무래도 iLabel적인 부분이 추가되어야 할 부분?!ㄴ
        decoder = models.decoder_dict['imap'](
            dim=dim, c_dim=0, color=True,
            hidden_size=256, skips=[], n_blocks=4, pos_embedding_method=pos_embedding_method
        )
    return decoder
