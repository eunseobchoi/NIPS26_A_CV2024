CREATE TABLE IF NOT EXISTS "audit_logs" (
  "id"              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  "user_id"         uuid NOT NULL REFERENCES "users"("id") ON DELETE SET NULL,
  "organization_id" uuid NOT NULL REFERENCES "organizations"("id") ON DELETE CASCADE,
  "action"          text NOT NULL,
  "method"          text NOT NULL,
  "path"            text NOT NULL,
  "route_params"    jsonb,
  "response_status" integer,
  "ip_address"      text,
  "user_agent"      text,
  "request_id"      text,
  "created_at"      timestamp NOT NULL DEFAULT now()
);

CREATE INDEX "audit_logs_organization_id_idx" ON "audit_logs" ("organization_id");
CREATE INDEX "audit_logs_user_id_idx"         ON "audit_logs" ("user_id");
CREATE INDEX "audit_logs_created_at_idx"      ON "audit_logs" ("created_at");
CREATE INDEX "audit_logs_action_idx"          ON "audit_logs" ("action");
