CREATE TABLE users (
  id INT AUTO_INCREMENT PRIMARY KEY,
  email VARCHAR(255) NOT NULL,
  password VARCHAR(255) NOT NULL
);

INSERT INTO users (email, password) VALUES ('test@example.com', 'password123');

CREATE TABLE orders (
  id INT AUTO_INCREMENT PRIMARY KEY,
  user_id INT NOT NULL,
  item VARCHAR(255) NOT NULL,
  amount DECIMAL(10, 2) NOT NULL
);
-- deliberately no index on orders.user_id, to compound the cross-join in /api/orders

-- seed a few thousand rows so the cross-join in /api/orders is actually slow
SET SESSION cte_max_recursion_depth = 5000;

INSERT INTO orders (user_id, item, amount)
WITH RECURSIVE seq(n) AS (
  SELECT 1
  UNION ALL
  SELECT n + 1 FROM seq WHERE n < 3000
)
SELECT 1, CONCAT('item-', n), ROUND(RAND() * 100, 2) FROM seq;
