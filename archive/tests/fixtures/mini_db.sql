-- Frozen mini fixture for ground-truth unit tests (INSERTs only; schema loaded from
-- specialist_router.env.database.schema_sql() to avoid duplicating the DDL).
--
-- These rows are hand-chosen so every task template's answer can be computed by hand; the
-- expected literals live in tests/unit/test_tasks_ground_truth.py. Do not edit without
-- recomputing those literals.

-- customers: (id, signup_date, country, segment, marketing_channel)
INSERT INTO customers VALUES (1, '2022-01-10', 'US', 'consumer', 'organic');
INSERT INTO customers VALUES (2, '2023-02-15', 'CA', 'consumer', 'paid_search');
INSERT INTO customers VALUES (3, '2022-06-01', 'GB', 'smb', 'referral');
INSERT INTO customers VALUES (4, '2023-03-03', 'DE', 'enterprise', 'email');

-- products: (id, name, category, price_cents, cost_cents, is_active)
INSERT INTO products VALUES (1, 'electronics-00001', 'electronics', 10000, 6000, 1);
INSERT INTO products VALUES (2, 'apparel-00002', 'apparel', 5000, 3000, 1);
INSERT INTO products VALUES (3, 'electronics-00003', 'electronics', 20000, 12000, 1);
INSERT INTO products VALUES (4, 'books-00004', 'books', 1000, 600, 0);

-- orders: (id, customer_id, order_ts, status, channel)
INSERT INTO orders VALUES (1, 1, '2022-03-01 10:00:00', 'completed', 'web');
INSERT INTO orders VALUES (2, 1, '2022-04-01 10:00:00', 'completed', 'web');
INSERT INTO orders VALUES (3, 2, '2023-05-01 10:00:00', 'completed', 'mobile');
INSERT INTO orders VALUES (4, 3, '2022-07-01 10:00:00', 'completed', 'partner');
INSERT INTO orders VALUES (5, 4, '2023-08-01 10:00:00', 'cancelled', 'web');
INSERT INTO orders VALUES (6, 2, '2023-06-01 10:00:00', 'completed', 'mobile');
INSERT INTO orders VALUES (7, 1, '2024-01-15 10:00:00', 'completed', 'web');

-- order_items: (id, order_id, product_id, quantity, unit_price_cents, discount_cents)
INSERT INTO order_items VALUES (1, 1, 1, 2, 10000, 500);
INSERT INTO order_items VALUES (2, 1, 2, 1, 5000, NULL);
INSERT INTO order_items VALUES (3, 2, 3, 1, 20000, 0);
INSERT INTO order_items VALUES (4, 3, 2, 3, 5000, 1000);
INSERT INTO order_items VALUES (5, 3, 4, 2, 1000, NULL);
INSERT INTO order_items VALUES (6, 4, 1, 1, 10000, 0);
INSERT INTO order_items VALUES (7, 5, 3, 1, 20000, 0);
INSERT INTO order_items VALUES (8, 6, 1, 5, 10000, 2000);
INSERT INTO order_items VALUES (9, 7, 4, 1, 1000, 100);

-- refunds: (id, order_id, refund_ts, amount_cents, reason)
INSERT INTO refunds VALUES (1, 1, '2022-03-05 09:00:00', 3000, 'defective');
INSERT INTO refunds VALUES (2, 3, '2023-05-10 09:00:00', 5000, 'changed_mind');
INSERT INTO refunds VALUES (3, 6, '2023-06-10 09:00:00', 1000, 'late_delivery');

-- returns: (id, order_item_id, return_ts, quantity_returned, reason)
INSERT INTO returns VALUES (1, 1, '2022-03-08 09:00:00', 1, 'wrong_size');
INSERT INTO returns VALUES (2, 8, '2023-06-12 09:00:00', 2, 'defective');
INSERT INTO returns VALUES (3, 4, '2023-05-12 09:00:00', 3, 'damaged');
