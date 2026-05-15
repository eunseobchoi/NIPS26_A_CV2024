import type { FastifyInstance, FastifyReply, FastifyRequest } from "fastify";
import { db } from "@/database";
import auditLogsTable from "@/database/schemas/audit-log";

/**
 * Map of mutating HTTP methods that should be recorded.
 * GET / HEAD / OPTIONS are intentionally excluded.
 */
const MUTATING_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);

/**
 * Derive a human-readable action label from the request.
 *
 * Examples:
 *   POST /api/agents             → "agent.create"
 *   PUT  /api/agents/:agentId    → "agent.update"
 *   DELETE /api/agents/:agentId  → "agent.delete"
 *   POST /api/llm-provider-api-keys → "llm-provider-api-key.create"
 */
function deriveAction(req: FastifyRequest): string {
  const parts = req.url.split("?")[0].split("/").filter(Boolean);
  // Remove the "api" prefix
  const apiIdx = parts.indexOf("api");
  const resourceParts = apiIdx >= 0 ? parts.slice(apiIdx + 1) : parts;

  if (resourceParts.length === 0) return "unknown";

  // Last segment may be an ID param — treat it as the target resource
  const last = resourceParts[resourceParts.length - 1];
  const isId = /^[0-9a-f-]{36}$/i.test(last) || /^\d+$/.test(last);

  const resource = isId && resourceParts.length > 1
    ? resourceParts.slice(0, -1).join(".")
    : resourceParts.join(".");

  const suffix = isId
    ? req.method === "DELETE"
      ? "delete"
      : "update"
    : req.method === "POST"
      ? "create"
      : req.method.toLowerCase();

  return `${resource}.${suffix}`;
}

/**
 * Extract client IP, respecting reverse-proxy headers.
 */
function getClientIp(req: FastifyRequest): string | null {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string") return forwarded.split(",")[0].trim();
  return req.ip ?? req.socket?.remoteAddress ?? null;
}

/**
 * Fastify plugin that hooks into onSend to record an audit log entry
 * for every authenticated mutating (POST/PUT/PATCH/DELETE) request.
 *
 * Safe by design:
 *  - Never blocks the response (fire-and-forget insert).
 *  - Does NOT store request bodies.
 *  - Skips unauthenticated requests and read-only methods.
 *  - Catches and logs DB errors without failing the request.
 */
export async function auditLoggerPlugin(app: FastifyInstance) {
  app.addHook(
    "onSend",
    async (req: FastifyRequest, reply: FastifyReply) => {
      try {
        // Only audit mutating requests
        if (!MUTATING_METHODS.has(req.method)) return;

        // Only audit authenticated users
        const userId = (req as any).session?.userId;
        const organizationId = (req as any).session?.organizationId;
        if (!userId || !organizationId) return;

        // Skip the audit-log routes themselves to avoid recursion
        if (req.url.startsWith("/api/audit-logs")) return;

        const entry = {
          userId,
          organizationId,
          action: deriveAction(req),
          method: req.method,
          path: req.url.split("?")[0],
          routeParams: (req.params as Record<string, unknown>) ?? {},
          responseStatus: reply.statusCode,
          ipAddress: getClientIp(req),
          userAgent: req.headers["user-agent"] ?? null,
          requestId: (req as any).id ?? null,
        };

        // Fire-and-forget — never await, never block the response
        db.insert(auditLogsTable)
          .values(entry)
          .catch((err: unknown) => {
            req.log.error({ err }, "Failed to write audit log entry");
          });
      } catch {
        // Swallow hook errors — never break a request because of auditing
      }
    },
  );
}

export default auditLoggerPlugin;
