-- Claim submitted content tasks atomically from backend-owned tables.
-- :claim_stale_after_sec int
-- :batch_size            int
-- :worker_id             text

WITH picked AS (
    SELECT
        ut.id,
        ut.user_id,
        ut.task_id,
        ut.external_ref,
        GREATEST(t.target, 1) AS target
    FROM user_tasks ut
    JOIN tasks t ON t.task_id = ut.task_id
    WHERE ut.status = 'submitted'
      AND t.type = 'content'
      AND NULLIF(BTRIM(ut.external_ref), '') IS NOT NULL
      AND (
          ut.progress_json IS NULL
          OR COALESCE(ut.progress_json->>'analysisStatus', 'pending') IN ('pending', 'queued')
          OR (
              ut.progress_json->>'analysisStatus' = 'processing'
              AND COALESCE(NULLIF(ut.progress_json->>'pickedAt', '')::timestamptz, NOW())
                    < NOW() - (:claim_stale_after_sec * INTERVAL '1 second')
          )
      )
    ORDER BY ut.completed_at NULLS FIRST, ut.updated_at, ut.id
    LIMIT :batch_size
    FOR UPDATE SKIP LOCKED
)
UPDATE user_tasks ut
SET progress_json = (
        CASE
            WHEN jsonb_typeof(ut.progress_json) = 'object' THEN ut.progress_json
            ELSE '{}'::jsonb
        END
        || jsonb_build_object(
            'analysisStatus', 'processing',
            'workerId', :worker_id,
            'pickedAt', NOW(),
            'attemptCount', COALESCE((ut.progress_json->>'attemptCount')::int, 0) + 1
        )
    ),
    updated_at = NOW()
FROM picked
WHERE ut.id = picked.id
RETURNING
    ut.id::text AS id,
    ut.user_id::text AS user_id,
    ut.task_id,
    ut.external_ref AS url,
    picked.target,
    COALESCE((ut.progress_json->>'attemptCount')::int, 0) AS attempt_count;
