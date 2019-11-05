# quantize_TF
量化训练及部署转tflite

#1. 量化训练。加在定义好loss之后，优化器之前
g = tf.get_default_graph()
tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=0)#首先需要重头训练，后面可以使用预训练

#2. 完成量化训练后，读取量化图并保存
x=tf.placeholder(tf.float32,shape=[None,128,128,3],name='x')
y_=tf.placeholder(tf.int32,shape=[None,11],name='y_')
is_training=tf.placeholder(tf.bool,name='is_train')
logits=nasnet(x,num_classes=11,is_train=is_training)
l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
#loss=focal_loss(logits,y_)+l2_loss
loss=tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=logits)+l2_loss
#loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)+l2_loss
g = tf.get_default_graph()
tf.contrib.quantize.create_eval_graph(input_graph=g)
eval_graph_file ='./models/quan/eval_graph_def.pb'
saver=tf.train.Saver()
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    #tf.reset_default_graph()
    #tf.contrib.quantize.create_eval_graph()
    with open(eval_graph_file, 'w') as f:
        f.write(str(g.as_graph_def()))
    saver.restore(sess,'./models/quan/model.ckpt')
    saver.save(sess, './models/quan/eval.ckpt')

#3. 冻结量化图.ckpt
def frozen():    
    input_node = tf.placeholder(tf.float32, shape=(1, 128, 128, 3), name="input") #这个是你送入网络的图片大小，如果你是其他的大小自行修改
    #input_node = tf.expand_dims(input_node, 0)
    flow = nasnet(input_node, 11)
    #flow = tf.cast(flow, tf.uint8, 'out') #设置输出类型以及输出的接口名字，为了之后的调用pb的时候使用
    #tf.contrib.quantize.create_eval_graph(input_graph=tf.get_default_graph())
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, model_path)
        #保存图
        tf.train.write_graph(sess.graph_def, './models/quan/', 'model.pb')
        #把图和参数结构一起
        freeze_graph.freeze_graph('./models/quan/model.pb', '', False, model_path, 'nas_tcl/logits/logits/Conv2D','save/restore_all', 'save/Const:0', './models/quan/frozen_model.pb', False, "")

    print("done")
    
#4. pb转tflite
def frozen2tflite():
    path_to_frozen_graphdef_pb = './models/quan/frozen_model.pb'
    #input_shapes = {'validate_input/imgs':[1,320,320,3]}
    #converter = tf.compat.v1.lite.TFLiteConverter.
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(path_to_frozen_graphdef_pb, ['input'], ['nas_tcl/logits/logits/Conv2D'], input_shapes={'input':[1,128,128,3]})
    #(tf_version<=1.11)converter = tf.contrib.lite.TocoConverter.from_frozen_graph(path_to_frozen_graphdef_pb, ['validate_input/imgs'], ['output_node'])
    converter.inference_type = tf.uint8
    #converter.default_ranges_stats = [127, 127]#representing the mean andstandard deviation.
    converter.quantized_input_stats = {'input':[0, 6]}#integers representing (min, max) range values
    converter.allow_custom_ops = True
    #converter.std_dev = 127
    #converter.mean = 127
    #converter.quantized_input_stats = {'x':(0.,1.)}#
    #converter.allow_custom_ops = True
    converter.default_ranges_stats = (0, 255)#这里需要研究一下
    #onverter.post_training_quantize = True
    tflite_model = converter.convert()
    open("./models/quan/nas_scene_q.tflite", "wb").write(tflite_model)
