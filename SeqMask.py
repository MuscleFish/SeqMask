from MaskAttention import *
from tensorflow.keras import layers,models,metrics,losses,optimizers,callbacks,activations

tact_ids = ['TA0001', 'TA0002', 'TA0003', 'TA0004', 'TA0005', 'TA0006',
            'TA0007', 'TA0008', 'TA0009', 'TA0010', 'TA0011', 'TA0040']
tech_ids = ['T1001', 'T1003', 'T1005', 'T1007', 'T1008', 'T1010', 'T1012', 'T1014', 'T1016', 'T1018', 'T1020', 'T1021',
            'T1025', 'T1027', 'T1029', 'T1030', 'T1033', 'T1036', 'T1037', 'T1039', 'T1040', 'T1041', 'T1046', 'T1047',
            'T1048', 'T1049', 'T1053', 'T1055', 'T1057', 'T1059', 'T1068', 'T1069', 'T1070', 'T1071', 'T1072', 'T1074',
            'T1078', 'T1080', 'T1082', 'T1083', 'T1087', 'T1090', 'T1091', 'T1092', 'T1095', 'T1098', 'T1102', 'T1104',
            'T1105', 'T1106', 'T1110', 'T1111', 'T1112', 'T1113', 'T1115', 'T1119', 'T1120', 'T1123', 'T1124', 'T1125',
            'T1129', 'T1132', 'T1133', 'T1134', 'T1135', 'T1137', 'T1140', 'T1176', 'T1185', 'T1187', 'T1189', 'T1190',
            'T1195', 'T1197', 'T1199', 'T1200', 'T1201', 'T1202', 'T1203', 'T1205', 'T1207', 'T1210', 'T1211', 'T1213',
            'T1217', 'T1219', 'T1220', 'T1221', 'T1398', 'T1400', 'T1401', 'T1402', 'T1404', 'T1406', 'T1407', 'T1409',
            'T1410', 'T1411', 'T1412', 'T1413', 'T1414', 'T1417', 'T1418', 'T1420', 'T1422', 'T1424', 'T1426', 'T1428',
            'T1429', 'T1430', 'T1432', 'T1433', 'T1435', 'T1436', 'T1437', 'T1438', 'T1444', 'T1446', 'T1447', 'T1448',
            'T1452', 'T1456', 'T1458', 'T1471', 'T1472', 'T1474', 'T1475', 'T1476', 'T1477', 'T1478', 'T1480', 'T1481',
            'T1482', 'T1484', 'T1485', 'T1486', 'T1489', 'T1490', 'T1496', 'T1497', 'T1498', 'T1499', 'T1507', 'T1508',
            'T1509', 'T1512', 'T1513', 'T1516', 'T1517', 'T1518', 'T1520', 'T1521', 'T1523', 'T1528', 'T1529', 'T1531',
            'T1532', 'T1533', 'T1534', 'T1539', 'T1540', 'T1541', 'T1543', 'T1544', 'T1552', 'T1554', 'T1555', 'T1556',
            'T1560', 'T1566', 'T1567', 'T1568', 'T1570', 'T1571', 'T1572', 'T1573', 'T1574', 'T1575', 'T1576', 'T1577',
            'T1579', 'T1581', 'T1582', 'T1585']

class PackCNNLayer(layers.Layer):
    def __init__(self, cd, cnn_count=256, **kwargs):
        super(PackCNNLayer, self).__init__(**kwargs)
        self.cx = layers.Conv1D(cnn_count, kernel_size=cd, padding='same')
        self.lx = layers.LayerNormalization()
        self.rx = layers.ReLU()
        self.gx = layers.GlobalMaxPooling1D()

    def call(self, x, trainable=None):
        cx = self.cx(x)
        lx = self.lx(cx)
        rx = self.rx(lx)
        gx = self.gx(rx)
        return gx


class PackDenseLayer(layers.Layer):
    def __init__(self, units, dropout=0.3, **kwargs):
        super(PackDenseLayer, self).__init__(**kwargs)
        self.dx = layers.Dense(units)
        self.bx = layers.BatchNormalization()
        self.rx = layers.ReLU()
        self.dpx = layers.Dropout(dropout)

    def call(self, x, trainable=None):
        dx = self.dx(x)
        bx = self.bx(dx)
        rx = self.rx(bx)
        dpx = self.dpx(rx)
        return dpx

def build_mask_attention(input_layer, mask_type="sv", mask_power=1., usedk=False,
                         mask_weight_layer_name="mask_weights_layer",
                         softmax_score_layer_name="softmax_score_layer"):
    use_mask = True
    if mask_type in ["s", "sv", "SV", "SimpleVector"]:
        x = SimpleVectorMask(name=mask_weight_layer_name)(input_layer)
    elif mask_type in ["m", "mp", "MP", "MiddlePoint"]:
        x = MiddlePointMaskWithBias(name=mask_weight_layer_name, power=mask_power)(input_layer)
    elif mask_type in ["a", "ar", "AR", "AreaRange"]:
        x = AreaRangeMaskWithBias(name=mask_weight_layer_name, power=mask_power)(input_layer)
    elif mask_type in ["mpnb", "MiddlePointNoBias"]:
        x = MiddlePointMaskWithoutBias(name=mask_weight_layer_name, power=mask_power)(input_layer)
    elif mask_type in ["arnb", "AreaRangeNoBias"]:
        x = AreaRangeMaskWithoutBias(name=mask_weight_layer_name, power=mask_power)(input_layer)
    elif mask_type in ["w", "wq", "WordsQuery"]:
        x = WordsQueryMask(name=mask_weight_layer_name, usedk=usedk)(input_layer)
    elif mask_type in ["e", "eq", "EmbeddingQuery"]:
        x = EmbeddingQueryMask(name=mask_weight_layer_name, usedk=usedk)(input_layer)
    else:
        use_mask=False
        x = input_layer

    if use_mask:
        x = layers.Softmax()(x)
        x = layers.Reshape((x.shape[1],1), name=softmax_score_layer_name)(x)
        mask_layer = x * input_layer
    else:
        mask_layer = x

    return mask_layer

class SeqMaskModel(models.Model):
    def __init__(self, input_shape, label_num=12, mask_type='sv', mask_power=None,
                 usedk=False, kernel_sizes=None, cnn_count=256, dense_dims=None,
                 dropout=.3, activation='sigmoid',
                 **kwargs):
        super(SeqMaskModel, self).__init__(**kwargs)
        if not mask_power:
            self.mask_power = 2. if mask_type in ["m", "mp", "MP", "MiddlePoint","mpnb", "MiddlePointNoBias"] else .5
        if not kernel_sizes or type(kernel_sizes) != list or len(kernel_sizes)==0:
            self.kernel_sizes = [1]
        else:
            self.kernel_sizes = kernel_sizes

        # build model
        self.input_layer = layers.Input(input_shape)
        self.multi_gram_layers = []
        for cni, ks in enumerate(self.kernel_sizes):
            cnn_layer = layers.Conv1D(cnn_count,
                                      kernel_size=ks,
                                      strides=1,
                                      padding='same',
                                      name="multi_gram_conv1d_"+str(cni))(self.input_layer)
            mask_score_layer = build_mask_attention(cnn_layer,
                                                    mask_type=mask_type,
                                                    mask_power=mask_power,
                                                    usedk=usedk,
                                                    mask_weight_layer_name='mask_weight_layer_4_cnn_'+str(cni),
                                                    softmax_score_layer_name='softmax_score_layer_4_cnn_'+str(cni)
                                                    )
            multiply_layer = tf.multiply(cnn_layer, mask_score_layer)
            max_layer = layers.GlobalMaxPool1D(multiply_layer)
            self.multi_gram_layers.append(max_layer)
        self.concatenate_layer = layers.Concatenate()(self.multi_gram_layers)
        self.flatten_layer = layers.Flatten()(self.concatenate_layer)
        dense_input_layer = self.flatten_layer
        for di, dense_dim in enumerate(dense_dims):
            dense_layer = layers.Dense(dense_dim)(dense_input_layer)
            bn_layer = layers.BatchNormalization()(dense_layer)
            activation_layer = layers.ReLU(bn_layer)
            dense_input_layer = layers.Dropout(dropout)(activation_layer)
        self.output_layer = layers.Dense(label_num, activation=activation)(dense_input_layer)

    def call(self, inputs, training=None, mask=None):
        return self.output_layer