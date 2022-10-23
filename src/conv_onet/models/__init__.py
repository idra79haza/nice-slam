from src.conv_onet.models import decoder

# Decoder dictionary
decoder_dict = {
    'nice': decoder.NICE,
    'imap':decoder.MLP,
    'ilabel_plus': decoder.ILABEL
    # TODO: 여기에 이제 'iLabel적인 부분이 추가되어야'
}