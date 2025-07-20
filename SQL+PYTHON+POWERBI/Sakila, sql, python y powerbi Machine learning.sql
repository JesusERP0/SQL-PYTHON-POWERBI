

-- 1.1. Listar las tablas disponibles (MySQL-specific)
SHOW TABLES;

-- 1.2. Inspeccionar la estructura de tablas clave
DESCRIBE customer;
DESCRIBE rental;
DESCRIBE payment;
DESCRIBE film;
DESCRIBE inventory;

-- 1.3. Ver las primeras filas de tablas clave para entender los datos
SELECT * FROM customer LIMIT 10;
SELECT * FROM rental LIMIT 10;
SELECT * FROM payment LIMIT 10;
SELECT * FROM film LIMIT 10;
SELECT * FROM inventory LIMIT 10;


-- -------------------------------------------------------------------
-- FASE 2 y 3: Ingeniería de Características y Consolidación con CTEs para la Vista
-- Objetivo: Unir todas las características y definir la variable objetivo
--           directamente dentro de la definición de la vista,
--           asegurando que todas las CTEs estén definidas
--           dentro del mismo bloque WITH de la vista.
-- -------------------------------------------------------------------

-- Creación de una Vista Permanente para el análisis de clientes
-- Esta vista contendrá todas las características y la variable objetivo 'is_vip',
-- además de las predicciones del modelo de ML que guardarás en 'customer_ml_predictions'.
CREATE OR REPLACE VIEW v_customer_analysis AS
WITH
    -- CTE 1: Calcular el total gastado por cada cliente
    CustomerTotalPayment AS (
        SELECT
            c.customer_id,
            SUM(p.amount) AS total_spent
        FROM
            customer c
        JOIN
            payment p ON c.customer_id = p.customer_id
        GROUP BY
            c.customer_id
    ),
    -- CTE 2: Calcular el número total de alquileres por cliente
    CustomerTotalRentals AS (
        SELECT
            c.customer_id,
            COUNT(r.rental_id) AS total_rentals
        FROM
            customer c
        JOIN
            rental r ON c.customer_id = r.customer_id
        GROUP BY
            c.customer_id
    ),
    -- CTE 3: Calcular el promedio de duración de alquiler por cliente (en días)
    CustomerAvgRentalDuration AS (
        SELECT
            c.customer_id,
            AVG(TIMESTAMPDIFF(HOUR, r.rental_date, r.return_date) / 24.0) AS avg_rental_duration_days
        FROM
            customer c
        JOIN
            rental r ON c.customer_id = r.customer_id
        WHERE
            r.return_date IS NOT NULL
        GROUP BY
            c.customer_id
    ),
    -- CTE 4: Obtener la fecha del último alquiler (Recencia)
    CustomerLastRentalDate AS (
        SELECT
            c.customer_id,
            MAX(r.rental_date) AS last_rental_date
        FROM
            customer c
        JOIN
            rental r ON c.customer_id = r.customer_id
        GROUP BY
            c.customer_id
    ),
    -- CTE 5: Obtener el número de películas distintas alquiladas por cliente
    CustomerUniqueFilms AS (
        SELECT
            c.customer_id,
            COUNT(DISTINCT i.film_id) AS unique_films_rented
        FROM
            customer c
        JOIN
            rental r ON c.customer_id = r.customer_id
        JOIN
            inventory i ON r.inventory_id = i.inventory_id
        GROUP BY
            c.customer_id
    ),
    -- CTE 6: Consolidar todas las características del cliente de Sakila
    CustomerFeatures AS (
        SELECT
            c.customer_id,
            c.first_name,
            c.last_name,
            c.email,
            c.active,
            a.address,
            a.district,
            ct.city,
            co.country,
            c.create_date AS customer_since_date,
            COALESCE(ctp.total_spent, 0) AS total_spent,
            COALESCE(ctr.total_rentals, 0) AS total_rentals,
            COALESCE(card.avg_rental_duration_days, 0) AS avg_rental_duration_days,
            COALESCE(clrd.last_rental_date, c.create_date) AS last_rental_date,
            COALESCE(cuf.unique_films_rented, 0) AS unique_films_rented
        FROM
            customer c
        LEFT JOIN
            address a ON c.address_id = a.address_id
        LEFT JOIN
            city ct ON a.city_id = ct.city_id
        LEFT JOIN
            country co ON ct.country_id = co.country_id
        LEFT JOIN
            CustomerTotalPayment ctp ON c.customer_id = ctp.customer_id
        LEFT JOIN
            CustomerTotalRentals ctr ON c.customer_id = ctr.customer_id
        LEFT JOIN
            CustomerAvgRentalDuration card ON c.customer_id = card.customer_id
        LEFT JOIN
            CustomerLastRentalDate clrd ON c.customer_id = clrd.customer_id
        LEFT JOIN
            CustomerUniqueFilms cuf ON c.customer_id = cuf.customer_id
    )
-- Consulta final que forma la vista, incluyendo la variable objetivo 'is_vip'
-- ¡IMPORTANTE!: customer_ml_predictions DEBE EXISTIR Y TENER DATOS
-- ANTES de ejecutar esta vista, si no, los campos de predicción serán NULL.
SELECT
    cf.*,
    CASE
        WHEN cf.total_spent > 150 THEN 1
        ELSE 0
    END AS is_vip,
    COALESCE(ml_pred.predicted_is_vip, -1) AS predicted_is_vip, -- Usamos -1 para indicar sin predicción si no hay match
    COALESCE(ml_pred.predicted_vip_proba, 0.0) AS predicted_vip_proba
FROM
    CustomerFeatures cf
LEFT JOIN
    customer_ml_predictions ml_pred ON cf.customer_id = ml_pred.customer_id
ORDER BY
    cf.customer_id;