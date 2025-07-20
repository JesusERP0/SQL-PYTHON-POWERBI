USE sakila;

Show Tables;
-- Test 1
--  Mostrar todos los actores
SELECT * FROM actor;
SELECT actor_id,first_name FROM actor;

--  Obtener nombres y apellidos de los clientes
SELECT first_name, last_name FROM customer;

--  Top 10 películas por título alfabéticamente
SELECT title FROM film ORDER BY title LIMIT 10;

--  Películas con duración mayor a 120 min
SELECT title, length FROM film WHERE length > 120;


-- Test 2 Agregación Filtrado
--  Número de películas por categoría
SELECT category.name, COUNT(*) AS total
FROM film
JOIN film_category ON film.film_id = film_category.film_id
JOIN category ON film_category.category_id = category.category_id
GROUP BY category.name;

--  Duración promedio por clasificación (rating)
SELECT rating, AVG(length) AS promedio_duracion
FROM film
GROUP BY rating;

--  Categorías con más de 70 películas(filtro)
SELECT category.name, COUNT(*) AS total
FROM film
JOIN film_category ON film.film_id = film_category.film_id
JOIN category ON film_category.category_id = category.category_id
GROUP BY category.name
HAVING COUNT(*) > 70;
-- Foreign e sports son las categorias que tienen mas de 70 peliculas

-- Test 3
--  Mostrar películas y su categoría
SELECT film.title, category.name AS categoria
FROM film
JOIN film_category ON film.film_id = film_category.film_id
JOIN category ON film_category.category_id = category.category_id;

--  Clientes y su dirección
SELECT customer.first_name, customer.last_name, address.address
FROM customer
JOIN address ON customer.address_id = address.address_id;

--  Alquileres con nombre de película y cliente
SELECT rental.rental_date, film.title, customer.first_name, customer.last_name
FROM rental
JOIN inventory ON rental.inventory_id = inventory.inventory_id
JOIN film ON inventory.film_id = film.film_id
JOIN customer ON rental.customer_id = customer.customer_id;

-- Test 4 LIMP TRANSF
--  Nombres de películas en mayúsculas
SELECT UPPER(title) AS titulo_mayuscula FROM film;
-- as crea nombre de columna 

--  Clasificar duración de películas
SELECT title,
  CASE
    WHEN length > 120 THEN 'Larga'
    WHEN length BETWEEN 90 AND 120 THEN 'Media'
    ELSE 'Corta'
  END AS duracion_clasificada
FROM film;

--  Extraer año del alquiler
-- SELECT rental_id, rental_date, STRFTIME('%Y', rental_date) AS año FROM rental;

-- Test 5 Subconsultas y CTEs
--  Películas más largas que el promedio
SELECT title, length
FROM film
WHERE length > (SELECT AVG(length) FROM film);

--  CTE: películas y su ranking por duración
WITH ranking_peliculas AS (
  SELECT title, length,
    RANK() OVER (ORDER BY length DESC) AS ranking
  FROM film
)
SELECT * FROM ranking_peliculas WHERE ranking <= 5;

-- Test 6 Analisis 
-- 1. Ranking de clientes por número de alquileres
SELECT customer_id, COUNT(*) AS total_alquileres,
  RANK() OVER (ORDER BY COUNT(*) DESC) AS ranking
FROM rental
GROUP BY customer_id;
-- ID 148 ES EL QUE MAS PELICULAS A ALQUILADO DESPUES LE SIGUE 526, ID 144 Y 236 EMPATAN EN EL 3ER PUESTO CON 42 ALQUILERES.  


-- 2. Alquileres acumulados por cliente en el tiempo
SELECT customer_id, rental_date, rental_id,
  COUNT(*) OVER (PARTITION BY customer_id ORDER BY rental_date) AS acumulado
FROM rental;





