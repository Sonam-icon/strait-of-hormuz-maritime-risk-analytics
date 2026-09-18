-- Strait of Hormuz Maritime Trade & Risk Analytics
-- Import the CSV into a table named hormuz_trade.

-- 1. Trade-tier summary
SELECT
    trade_tier,
    COUNT(*) AS records,
    COUNT(DISTINCT mmsi) AS unique_vessels,
    AVG(toll_usd) AS avg_toll_usd,
    AVG(insurance_cost_usd) AS avg_insurance_usd,
    AVG(days_delayed) AS avg_delay_days,
    AVG(total_transit_cost_usd) AS avg_total_transit_cost_usd
FROM hormuz_trade
GROUP BY trade_tier;


-- 2. Rerouting rate by continent
SELECT
    continent,
    COUNT(*) AS total_records,
    SUM(
        CASE
            WHEN rerouted = 'Yes (Cape of Good Hope)' THEN 1
            ELSE 0
        END
    ) AS rerouted_records,
    ROUND(
        100.0 * SUM(
            CASE
                WHEN rerouted = 'Yes (Cape of Good Hope)' THEN 1
                ELSE 0
            END
        ) / COUNT(*),
        2
    ) AS reroute_rate_pct
FROM hormuz_trade
GROUP BY continent
ORDER BY reroute_rate_pct DESC;


-- 3. Cost by commodity
SELECT
    commodity,
    COUNT(*) AS records,
    AVG(total_transit_cost_usd) AS avg_transit_cost_usd,
    AVG(days_delayed) AS avg_delay_days,
    SUM(total_asset_value_at_risk_usd) AS total_asset_risk_usd
FROM hormuz_trade
GROUP BY commodity
ORDER BY avg_transit_cost_usd DESC;


-- 4. Payment-rail analysis
SELECT
    payment_rail,
    COUNT(*) AS records,
    AVG(total_transit_cost_usd) AS avg_transit_cost_usd,
    AVG(days_delayed) AS avg_delay_days,
    SUM(total_asset_value_at_risk_usd) AS total_asset_risk_usd
FROM hormuz_trade
GROUP BY payment_rail
ORDER BY avg_transit_cost_usd DESC;


-- 5. Daily operational metrics
SELECT
    date,
    COUNT(*) AS vessel_records,
    COUNT(DISTINCT mmsi) AS unique_vessels,
    AVG(days_delayed) AS avg_delay_days,
    SUM(total_transit_cost_usd) AS total_transit_cost_usd,
    SUM(total_asset_value_at_risk_usd) AS total_asset_risk_usd
FROM hormuz_trade
GROUP BY date
ORDER BY date;


-- 6. Asset-value reconciliation
SELECT
    COUNT(*) AS inconsistent_records
FROM hormuz_trade
WHERE total_asset_value_at_risk_usd <>
      ship_hull_value_usd + estimated_cargo_value_usd;


-- 7. Transit-cost reconciliation
SELECT
    COUNT(*) AS inconsistent_records
FROM hormuz_trade
WHERE total_transit_cost_usd <>
      toll_usd + insurance_cost_usd + reroute_penalty_usd;
