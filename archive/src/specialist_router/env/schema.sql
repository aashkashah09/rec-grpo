-- Seeded e-commerce schema for the Specialist + Router environment (Phase 1).
--
-- Design notes (see docs/decisions/001-environment-and-schema.md):
--   * All monetary values are stored as INTEGER cents to avoid floating-point drift;
--     answers are converted to USD dollars only at the verifier boundary.
--   * Dates/timestamps are ISO-8601 TEXT ('YYYY-MM-DD' / 'YYYY-MM-DD HH:MM:SS') so the
--     database is deterministic and diff-friendly across platforms.
--   * order_items.discount_cents is intentionally NULLABLE: NULL means "no discount
--     recorded", which the null_discount_edge task template exercises (NULL treated as 0).
--   * Foreign keys are declared and enforced (PRAGMA foreign_keys = ON at connect time).

CREATE TABLE customers (
    customer_id       INTEGER PRIMARY KEY,
    signup_date       TEXT    NOT NULL,           -- 'YYYY-MM-DD'
    country           TEXT    NOT NULL,
    segment           TEXT    NOT NULL,           -- consumer | smb | enterprise
    marketing_channel TEXT    NOT NULL
);

CREATE TABLE products (
    product_id  INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,
    price_cents INTEGER NOT NULL,
    cost_cents  INTEGER NOT NULL,
    is_active   INTEGER NOT NULL                  -- 0 | 1
);

CREATE TABLE orders (
    order_id    INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    order_ts    TEXT    NOT NULL,                 -- 'YYYY-MM-DD HH:MM:SS'
    status      TEXT    NOT NULL,                 -- completed | cancelled | pending
    channel     TEXT    NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE TABLE order_items (
    order_item_id    INTEGER PRIMARY KEY,
    order_id         INTEGER NOT NULL,
    product_id       INTEGER NOT NULL,
    quantity         INTEGER NOT NULL,
    unit_price_cents INTEGER NOT NULL,            -- price frozen at order time
    discount_cents   INTEGER,                     -- NULLABLE: NULL == no discount recorded
    FOREIGN KEY (order_id) REFERENCES orders (order_id),
    FOREIGN KEY (product_id) REFERENCES products (product_id)
);

CREATE TABLE refunds (
    refund_id   INTEGER PRIMARY KEY,
    order_id    INTEGER NOT NULL,
    refund_ts   TEXT    NOT NULL,                 -- 'YYYY-MM-DD HH:MM:SS'
    amount_cents INTEGER NOT NULL,                -- order-level monetary refund
    reason      TEXT    NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE TABLE returns (
    return_id         INTEGER PRIMARY KEY,
    order_item_id     INTEGER NOT NULL,
    return_ts         TEXT    NOT NULL,           -- 'YYYY-MM-DD HH:MM:SS'
    quantity_returned INTEGER NOT NULL,           -- item-level physical return (units)
    reason            TEXT    NOT NULL,
    FOREIGN KEY (order_item_id) REFERENCES order_items (order_item_id)
);

CREATE INDEX idx_orders_customer ON orders (customer_id);
CREATE INDEX idx_order_items_order ON order_items (order_id);
CREATE INDEX idx_order_items_product ON order_items (product_id);
CREATE INDEX idx_refunds_order ON refunds (order_id);
CREATE INDEX idx_returns_item ON returns (order_item_id);
