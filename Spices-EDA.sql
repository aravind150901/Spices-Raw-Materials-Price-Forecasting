CREATE DATABASE spicesdb;

SELECT * FROM spicesdb.spices;

DESCRIBE spicesdb.spices;
SELECT COUNT(*) FROM spicesdb.spices;

SELECT
    AVG(Price) AS mean,
    MIN(Price) AS min_value,
    MAX(Price) AS max_value,
    COUNT(*) AS total_count
FROM spicesdb.spices;

SELECT
    Spices,
    AVG(Price) AS avg_value,
    COUNT(*) AS count
FROM spicesdb.spices
GROUP BY Spices;

SELECT
    COUNT(CASE WHEN Price IS NULL THEN 1 END) AS null_count,
    COUNT(*) AS total_count
FROM spicesdb.spices;