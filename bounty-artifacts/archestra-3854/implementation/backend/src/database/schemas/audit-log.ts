import {
  index,
  integer,
  jsonb,
  pgTable,
  text,
  timestamp,
  uuid,
} from "drizzle-orm/pg-core";
import organizationsTable from "./organization";
import usersTable from "./user";

/**
 * Audit log table — records every authenticated mutating API request
 * for admin accountability and compliance.
 *
 * Intentionally does NOT store request bodies, which may contain secrets.
 */
const auditLogsTable = pgTable(
  "audit_logs",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    userId: uuid("user_id")
      .notNull()
      .references(() => usersTable.id, { onDelete: "set null" }),
    organizationId: uuid("organization_id")
      .notNull()
      .references(() => organizationsTable.id, { onDelete: "cascade" }),
    /** High-level action label inferred from the route, e.g. "agent.create" */
    action: text("action").notNull(),
    /** HTTP method (POST / PUT / PATCH / DELETE) */
    method: text("method").notNull(),
    /** Request path, e.g. "/api/agents" */
    path: text("path").notNull(),
    /** Captured route params (e.g. { agentId: "..." }) */
    routeParams: jsonb("route_params").$type<Record<string, unknown>>(),
    /** HTTP response status code */
    responseStatus: integer("response_status"),
    /** Client IP address (from X-Forwarded-For or socket) */
    ipAddress: text("ip_address"),
    /** User-Agent header */
    userAgent: text("user_agent"),
    /** Unique request ID generated per request */
    requestId: text("request_id"),
    createdAt: timestamp("created_at", { mode: "date" }).notNull().defaultNow(),
  },
  (table) => [
    index("audit_logs_organization_id_idx").on(table.organizationId),
    index("audit_logs_user_id_idx").on(table.userId),
    index("audit_logs_created_at_idx").on(table.createdAt),
    index("audit_logs_action_idx").on(table.action),
  ],
);

export default auditLogsTable;
