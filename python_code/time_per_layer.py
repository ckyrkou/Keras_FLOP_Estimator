#                    __
#                  / *_)
#       _.----. _ /../
#     /............./
# __/..(...|.(...|
# /__.-|_|--|_|
#
# Christos Kyrkou, PhD
# 2019

# Measure time per layer in keras

def time_per_layer(model):
    new_model = model

    times = np.zeros((len(model.layers), 2)
    inp = np.ones((240, 240, 3))

    for i in range(1, len(model.layers)):
        new_model = keras.models.Model(inputs=[model.input], outputs=[model.layers[-i].output])

        # new_model.summary()
        new_model.predict(inp[None, :, :, :])

        t_s = time.time()
        new_model.predict(inp[None, :, :, :])
        t_e2 = time.time() - t_s

        times[i, 1] = t_e2
        del new_model

    for i in range(0, len(model.layers) - 1):
        times[i, 0] = abs(times[i + 1, 1] - times[i, 1])

    times[-1, 0] = times[-1, 1]


    return times
