
!git clone https://github.com/ipeirotis/autoencoders_census.git

# Commented out IPython magic to ensure Python compatibility.
# %cd autoencoders_census

"""# Model Setup"""

latent_dimension = 1
batch_size = 20

hidden_nodes = 16

# The dimensionality of the dataframe is (nrows x dim).
# We keep the dim as the size of the input
input_dim = transformed_df.shape[1]

input_encoder = Input(shape=(input_dim,), name="Input_Encoder")

batch_normalize1 = BatchNormalization()(input_encoder)

hidden_layer = Dense(hidden_nodes, activation="relu", name="Hidden_Encoding")(
    batch_normalize1
)
batch_normalize2 = BatchNormalization()(hidden_layer)

z = Dense(latent_dimension, name="Mean")(batch_normalize2)

encoder = Model(input_encoder, z, name="Encoder")

input_decoder = Input(shape=(latent_dimension,), name="Input_Decoder")
batch_normalize1 = BatchNormalization()(input_decoder)

decoder_hidden_layer = Dense(hidden_nodes, activation="relu", name="Hidden_Decoding")(
    batch_normalize1
)
batch_normalize2 = BatchNormalization()(decoder_hidden_layer)

decoded = Dense(input_dim, activation="linear", name="Decoded")(batch_normalize2)

decoder = Model(input_decoder, decoded, name="Decoder")

encoder_decoder = decoder(encoder(input_encoder))

ae = Model(input_encoder, encoder_decoder)
ae.save("model_autoencoder_basic.h5")

"""## Model Training"""

def masked_mse(y_true, y_pred):
    mask = tf.where(tf.math.is_nan(y_true), 0.0, 1.0)
    return tf.reduce_mean(tf.square(y_true - y_pred) * mask)

ae.compile(loss=masked_mse, optimizer="adam", weighted_metrics=[])

# Replace null values in the dataframe with zeros
transformed_df = transformed_df.fillna(0.0)

mask = np.where(transformed_df.isnull(), 0.0, 1.0)
mask = np.expand_dims(mask, axis=-1)

# Initialize lists to track losses
train_loss = []
val_loss = []

class LossTracker(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss.append(logs.get('loss'))
        val_loss.append(logs.get('val_loss'))

loss_tracker = LossTracker()

history = ae.fit(
    transformed_df, transformed_df, sample_weight=mask, shuffle=True, epochs=10, batch_size=20,
    validation_split=0.2, verbose=0, callbacks=[loss_tracker]
)

"""# Hyperparameter Tuning"""

transformed_df = transformed_df.fillna(0.0)
X_train, X_test = train_test_split(transformed_df, test_size=0.2, random_state=42)

input_shape = X_train.shape[1:]
d = 5

# Define the encoder model
def build_encoder(hp):
    inputs = Input(shape=input_shape)
    x = Dense(units=hp.Int('encoder_units_1', min_value=32, max_value=256, step=32), activation='relu')(inputs)
    x = Dense(units=hp.Int('encoder_units_2', min_value=16, max_value=128, step=16), activation='relu')(x)
    latent_space = Dense(units=d, activation='relu')(x)
    encoder = Model(inputs, latent_space)
    return encoder

# Define the decoder model
def build_decoder(hp):
    decoder_inputs = Input(shape=(d,))
    x = Dense(units=hp.Int('decoder_units_1', min_value=16, max_value=128, step=16), activation='relu')(decoder_inputs)
    x = Dense(units=hp.Int('decoder_units_2', min_value=32, max_value=256, step=32), activation='relu')(x)
    outputs = Dense(units=input_shape[0], activation='linear')(x)
    decoder = Model(decoder_inputs, outputs)
    return decoder

# Define the autoencoder model by combining the encoder and decoder models
def build_autoencoder(hp):
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    batch_size = hp.Int('batch_size', min_value=16, max_value=128, step=16)
    autoencoder_input = Input(shape=input_shape)
    encoder_output = build_encoder(hp)(autoencoder_input)
    decoder_output = build_decoder(hp)(encoder_output)
    autoencoder = Model(autoencoder_input, decoder_output)
    autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
    return autoencoder

# Define the tuner
tuner = RandomSearch(
    build_autoencoder,
    objective='val_loss',
    max_trials=10,
    executions_per_trial=3,
    directory='my_dir',
    project_name='HDHyperparameter',
    overwrite=True,
    seed=42)

# Perform hyperparameter search
tuner.search(X_train, X_train, epochs=10, batch_size=32, validation_data=(X_test, X_test))

# Get the best hyperparameters and build the final model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
final_model = build_autoencoder(best_hps)

train_loss1 = []
val_loss1 = []

# Train the final model
history1 = final_model.fit(X_train, X_train,
                          epochs=10,
                          batch_size=32,
                          verbose=1,
                          validation_data=(X_test, X_test))

train_loss1.append(history1.history['loss'][-1])
val_loss1.append(history1.history['val_loss'][-1])

# Save the model as a .h5 file
model_filename = "Hyperparameter.h5"
final_model.save(model_filename)

# Store the hyperparameters and evaluation metrics in a dictionary
hyperparameters_dict = {"learning_rate": best_hps.get('learning_rate'),
                        "batch_size": best_hps.get('batch_size'),
                        "num_epochs": 10,
                        "loss": history.history['loss'][-1],
                        "val_loss": history.history['val_loss'][-1]}
