
MAPPING = {1:'gaussian-blur', 2:'shear', 3: 'horizontal-flip', 4: 'vertical-flip', 5: 'sharpen', 6: 'emboss',
       7: 'additive-gaussian-noise', 8: 'dropout', 9: 'coarse-dropout', 10: 'gamma-contrast', 11: 'brighten',
       12: 'invert', 13: 'fog', 14: 'clouds', 15: 'super-pixels', 16: 'elastic-transform', 17: 'grayscale'}


def parse_aug_param(params):
    policy = dict()

    if params['padding_value'] == 0:
        policy['padding_value'] = (0, 0, 0)
    elif params['padding_value'] == 128:
        policy['padding_value'] = (128, 128, 128)
    else:
        raise ValueError('Padding value should only be 0 or 128')

    for i in range(9):
        type_key = 'aug_{}'.format(i)
        aug_type = params[type_key]
        if aug_type != 'none':
            mag_key = 'magnitude_{}'.format(i)
            magnitude = params[mag_key]

            policy[aug_type] = magnitude
