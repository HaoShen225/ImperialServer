WITH vendor_stats AS (
    SELECT
        protocol,
        lr,
        vendor,
        AVG(dice) AS dice_mean,
        SQRT(
            MAX(
                0.0,
                (SUM(dice * dice) - SUM(dice) * SUM(dice) / COUNT(*))
                / (COUNT(*) - 1)
            )
        ) AS dice_sd,
        AVG(hd95) AS hd95_mean,
        SQRT(
            MAX(
                0.0,
                (SUM(hd95 * hd95) - SUM(hd95) * SUM(hd95) / COUNT(*))
                / (COUNT(*) - 1)
            )
        ) AS hd95_sd
    FROM grata_summary_metrics
    GROUP BY protocol, lr, vendor
),
seed_cross_vendor AS (
    SELECT
        protocol,
        lr,
        seed,
        AVG(dice) AS dice,
        AVG(hd95) AS hd95
    FROM grata_summary_metrics
    GROUP BY protocol, lr, seed
),
cross_vendor_stats AS (
    SELECT
        protocol,
        lr,
        AVG(dice) AS dice_mean,
        SQRT(
            MAX(
                0.0,
                (SUM(dice * dice) - SUM(dice) * SUM(dice) / COUNT(*))
                / (COUNT(*) - 1)
            )
        ) AS dice_sd,
        AVG(hd95) AS hd95_mean,
        SQRT(
            MAX(
                0.0,
                (SUM(hd95 * hd95) - SUM(hd95) * SUM(hd95) / COUNT(*))
                / (COUNT(*) - 1)
            )
        ) AS hd95_sd
    FROM seed_cross_vendor
    GROUP BY protocol, lr
)
SELECT
    cross.protocol,
    cross.lr,
    b.dice_mean AS vendor_b_dice_mean,
    b.dice_sd AS vendor_b_dice_sd,
    c.dice_mean AS vendor_c_dice_mean,
    c.dice_sd AS vendor_c_dice_sd,
    d.dice_mean AS vendor_d_dice_mean,
    d.dice_sd AS vendor_d_dice_sd,
    cross.dice_mean AS avg_dice_mean,
    cross.dice_sd AS avg_dice_sd,
    cross.hd95_mean AS avg_hd95_mean,
    cross.hd95_sd AS avg_hd95_sd
FROM cross_vendor_stats AS cross
JOIN vendor_stats AS b
  ON b.protocol = cross.protocol AND b.lr = cross.lr AND b.vendor = 'B'
JOIN vendor_stats AS c
  ON c.protocol = cross.protocol AND c.lr = cross.lr AND c.vendor = 'C'
JOIN vendor_stats AS d
  ON d.protocol = cross.protocol AND d.lr = cross.lr AND d.vendor = 'D'
ORDER BY
    CASE cross.protocol WHEN 'patient_volume' THEN 0 ELSE 1 END,
    cross.lr;
