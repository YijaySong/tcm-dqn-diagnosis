import pickle
import tensorflow as tf
dataset = "YAGO3-10"
relation = "influences"

# 计算QRL参数量
with open('models/' + 'quantum_model_' + dataset+'_' + relation.replace("/", "@") + '_retrained.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    # param_count = loaded_model.count_params()
    model_weights = loaded_model['weights']
    param_count = sum(tf.size(weight).numpy() for weight in model_weights)
    model_biases = loaded_model['biases']
    param_count += sum(tf.size(biase).numpy() for biase in model_biases)
    print(f"模型参数总量: {param_count}")    # 160

# RL参数量:1933874