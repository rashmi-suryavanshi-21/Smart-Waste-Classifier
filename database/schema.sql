CREATE TABLE waste_classes (
    class_id     INTEGER PRIMARY KEY,
    class_name   VARCHAR(20) NOT NULL UNIQUE,
    is_recyclable INTEGER NOT NULL      -- 1 = recyclable, 0 = nahi
);

CREATE TABLE images (
    image_id     INTEGER PRIMARY KEY,
    file_name    VARCHAR(255) NOT NULL,
    actual_class_id INTEGER NOT NULL,
    split_name   VARCHAR(10) NOT NULL,   -- 'test'
    FOREIGN KEY (actual_class_id) REFERENCES waste_classes(class_id)
);

CREATE TABLE predictions (
    prediction_id INTEGER PRIMARY KEY,
    image_id      INTEGER NOT NULL,
    predicted_class_id INTEGER NOT NULL,
    confidence    DECIMAL(5,4) NOT NULL,
    is_correct    INTEGER NOT NULL,      -- 1 = sahi, 0 = galat
    model_name    VARCHAR(30) NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(image_id),
    FOREIGN KEY (predicted_class_id) REFERENCES waste_classes(class_id)
);