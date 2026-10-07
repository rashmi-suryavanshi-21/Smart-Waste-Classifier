-- Har class ki accuracy
SELECT
    c.class_name,
    COUNT(*)                                   AS total_images,
    SUM(p.is_correct)                          AS correct,
    ROUND(100.0 * SUM(p.is_correct) / COUNT(*), 1) AS accuracy_pct
FROM images i
JOIN predictions p    ON p.image_id = i.image_id
JOIN waste_classes c  ON c.class_id = i.actual_class_id
GROUP BY c.class_name
ORDER BY accuracy_pct ASC;

-- Query 2: sabse common galtiyan (asli class -> model ka jawab)
SELECT
    a.class_name AS actual_class,
    pr.class_name AS predicted_class,
    COUNT(*) AS times
FROM images i
JOIN predictions p      ON p.image_id = i.image_id
JOIN waste_classes a    ON a.class_id = i.actual_class_id
JOIN waste_classes pr   ON pr.class_id = p.predicted_class_id
WHERE p.is_correct = 0
GROUP BY a.class_name, pr.class_name
ORDER BY times DESC;

-- Query 3: confidence ke hisaab se accuracy
SELECT
    CASE
        WHEN p.confidence < 0.5 THEN '1. low (<0.5)'
        WHEN p.confidence < 0.8 THEN '2. medium (0.5-0.8)'
        ELSE '3. high (>=0.8)'
    END AS confidence_band,
    COUNT(*)                                       AS total_images,
    SUM(p.is_correct)                              AS correct,
    ROUND(100.0 * SUM(p.is_correct) / COUNT(*), 1) AS accuracy_pct
FROM predictions p
GROUP BY confidence_band
ORDER BY confidence_band;