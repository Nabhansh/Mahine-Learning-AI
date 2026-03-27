"""
Handwritten Digit Recognition (MNIST)
CNN-based classifier with full training pipeline + live drawing demo.
Install: pip install tensorflow keras numpy matplotlib scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# ── Try TensorFlow/Keras, fall back to sklearn for zero-dependency demo ───────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    USE_KERAS = True
except ImportError:
    USE_KERAS = False
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    print("TensorFlow not found — using sklearn RandomForest on digits dataset.")

# ══════════════════════════════════════════════════════════════════════════════
if USE_KERAS:
    # ── 1. Load MNIST ─────────────────────────────────────────────────────────
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train.astype('float32') / 255.0
    X_test  = X_test.astype('float32')  / 255.0
    # Add channel dim → (28, 28, 1)
    X_train = X_train[..., np.newaxis]
    X_test  = X_test[...,  np.newaxis]
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # ── 2. Build CNN ──────────────────────────────────────────────────────────
    model = keras.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax'),
    ], name='MNIST_CNN')
    model.summary()

    # ── 3. Compile & Train ────────────────────────────────────────────────────
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1),
    ]
    # Data augmentation
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1
    )
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=128),
        epochs=20, validation_split=0.1,
        callbacks=callbacks, verbose=1,
    )

    # ── 4. Evaluate ───────────────────────────────────────────────────────────
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test).argmax(axis=1)
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    HISTORY = history.history
    test_images, test_labels, preds = X_test, y_test, y_pred

# ── sklearn fallback ─────────────────────────────────────────────────────────
else:
    data = load_digits()
    X, y = data.images, data.target
    X_flat = X.reshape(len(X), -1)
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = (preds == y_test).mean()
    print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, preds))

    HISTORY = None
    test_images = data.images[len(data.images) - len(y_test):]
    test_labels, preds = y_test, preds

# ── 5. Visualizations ────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Handwritten Digit Recognition', fontsize=16, fontweight='bold')

# Sample predictions
ax_samples = [fig.add_subplot(3, 10, i+1) for i in range(20)]
for i, ax in enumerate(ax_samples):
    img = test_images[i].squeeze() if test_images[i].ndim == 3 else test_images[i].reshape(8, 8)
    ax.imshow(img, cmap='gray')
    label = test_labels[i]
    pred  = preds[i]
    ax.set_title(f'P:{pred}', color='green' if pred == label else 'red', fontsize=9)
    ax.axis('off')

# Confusion matrix
cm = confusion_matrix(test_labels[:500] if len(test_labels) > 500 else test_labels,
                      preds[:500]      if len(preds)      > 500 else preds)
ax_cm = fig.add_subplot(3, 3, 4)
im = ax_cm.imshow(cm, cmap='Blues')
ax_cm.set_title('Confusion Matrix'); ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax_cm.text(j, i, cm[i,j], ha='center', va='center', fontsize=7,
                   color='white' if cm[i,j] > cm.max()/2 else 'black')

# Training curves (Keras only)
if HISTORY:
    ax_acc  = fig.add_subplot(3, 3, 5)
    ax_loss = fig.add_subplot(3, 3, 6)
    ax_acc.plot(HISTORY['accuracy'],     label='Train')
    ax_acc.plot(HISTORY['val_accuracy'], label='Val')
    ax_acc.set_title('Accuracy'); ax_acc.legend()
    ax_loss.plot(HISTORY['loss'],     label='Train')
    ax_loss.plot(HISTORY['val_loss'], label='Val')
    ax_loss.set_title('Loss'); ax_loss.legend()

# Per-digit accuracy
per_digit_acc = [
    (preds[test_labels == d] == d).mean()
    for d in range(10)
]
ax_bar = fig.add_subplot(3, 3, 7)
bars = ax_bar.bar(range(10), per_digit_acc, color=plt.cm.tab10.colors)
ax_bar.set_xticks(range(10)); ax_bar.set_ylim(0.8, 1.01)
ax_bar.set_title('Per-Digit Accuracy'); ax_bar.set_xlabel('Digit')
for bar, v in zip(bars, per_digit_acc):
    ax_bar.text(bar.get_x()+bar.get_width()/2, v+0.002, f'{v:.2f}', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('digit_recognition_results.png', dpi=150)
print("📊 Saved: digit_recognition_results.png")

# ── 6. Save model ─────────────────────────────────────────────────────────────
if USE_KERAS:
    model.save('mnist_cnn_model.keras')
    print("💾 Model saved: mnist_cnn_model.keras")
    print(f"\n✅ Final accuracy: {acc*100:.2f}%")

# ── 7. Inference helper ───────────────────────────────────────────────────────
def predict_digit(image_28x28: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Predict digit from a 28×28 grayscale array (0-255).
    Returns (predicted_class, confidence_array)
    """
    if not USE_KERAS:
        flat = image_28x28.flatten().reshape(1, -1)
        return model.predict(flat)[0], np.array([])
    img = image_28x28.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)
    probs = model.predict(img, verbose=0)[0]
    return int(probs.argmax()), probs

# Quick demo
sample_img = (test_images[5].squeeze() * 255).astype(np.uint8) if USE_KERAS else test_images[5].reshape(8, 8)
if USE_KERAS:
    digit, confidence = predict_digit(sample_img)
    print(f"\n[Demo] Predicted: {digit} | Confidence: {confidence.max()*100:.1f}%")
