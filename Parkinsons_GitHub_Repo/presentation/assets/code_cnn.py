
# CNN training with participant-aware sample weights and group-wise validation

cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_cnn_train.shape[1:]),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.45),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.35),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
    loss='binary_crossentropy',
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Give each participant equal total weight, then balance the two classes.
train_participant_counts = cnn_train_meta['participant'].value_counts()
n_train_participants = cnn_train_meta['participant'].nunique()
n_classes = cnn_train_meta['label'].nunique()

sample_weights = np.array([
    1.0 / train_participant_counts[pid]
    for pid in cnn_train_meta['participant']
], dtype=np.float32)

# Normalize weights so their mean is 1.0.
sample_weights *= len(sample_weights) / sample_weights.sum()

class_participant_counts = (
    cnn_train_meta[['participant', 'label']]
    .drop_duplicates()['label'].value_counts().to_dict()
)
class_weights_participant = {
    cls: n_train_participants / (n_classes * count)
    for cls, count in class_participant_counts.items()
}
sample_weights *= cnn_train_meta['label'].map(class_weights_participant).to_numpy()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=8, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
    )
]

history = cnn.fit(
    X_cnn_train, y_cnn_train,
    sample_weight=sample_weights,
    validation_data=(X_cnn_val, y_cnn_val),
    epochs=60,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

pd.DataFrame(history.history).to_csv(
    OUTPUT_DIR / 'cnn_training_history.csv', index=False
)
