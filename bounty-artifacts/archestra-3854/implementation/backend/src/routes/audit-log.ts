import { and, desc, eq, gte, ilike, lte, sql } from "drizzle-orm";
import type { FastifyInstance } from "fastify";
import { db } from "@/database";
import auditLogsTable from "@/database/schemas/audit-log";
import usersTable from "@/database/schemas/user";

/**
 * Audit-log API routes.
 *
 * All endpoints are admin-only (enforced by RBAC middleware).
 *
 * GET /api/audit-logs — paginated, filterable, sortable list.
 *
 * Query parameters:
 *   page       — 1-indexed page number (default 1)
 *   pageSize   — items per page (default 50, max 200)
 *   sortField  — "createdAt" | "action" | "method" | "responseStatus" | "userName" (default "createdAt")
 *   sortOrder  — "asc" | "desc" (default "desc")
 *   search     — free-text search against user name, path, action, and IP
 *   userId     — filter by user UUID
 *   method     — filter by HTTP method (POST/PUT/PATCH/DELETE)
 *   status     — filter by response status code
 *   dateFrom   — ISO date string, inclusive lower bound
 *   dateTo     — ISO date string, inclusive upper bound
 */
export async function auditLogRoutes(app: FastifyInstance) {
  app.get("/api/audit-logs", {
    preHandler: [(app as any).requireAdmin],
    handler: async (req, reply) => {
      const q = req.query as Record<string, string | undefined>;

      const page = Math.max(1, parseInt(q.page ?? "1", 10));
      const pageSize = Math.min(200, Math.max(1, parseInt(q.pageSize ?? "50", 10)));
      const offset = (page - 1) * pageSize;

      const sortField = q.sortField ?? "createdAt";
      const sortOrder = q.sortOrder ?? "desc";

      // Build WHERE conditions
      const conditions = [];

      // Scope to user's organization
      const organizationId = (req as any).session?.organizationId;
      if (organizationId) {
        conditions.push(eq(auditLogsTable.organizationId, organizationId));
      }

      if (q.userId) {
        conditions.push(eq(auditLogsTable.userId, q.userId));
      }

      if (q.method) {
        conditions.push(eq(auditLogsTable.method, q.method.toUpperCase()));
      }

      if (q.status) {
        conditions.push(eq(auditLogsTable.responseStatus, parseInt(q.status, 10)));
      }

      if (q.dateFrom) {
        conditions.push(gte(auditLogsTable.createdAt, new Date(q.dateFrom)));
      }

      if (q.dateTo) {
        conditions.push(lte(auditLogsTable.createdAt, new Date(q.dateTo)));
      }

      if (q.search) {
        const pattern = `%${q.search}%`;
        conditions.push(
          sql`(
            ${auditLogsTable.action} ILIKE ${pattern} OR
            ${auditLogsTable.path} ILIKE ${pattern} OR
            ${auditLogsTable.ipAddress} ILIKE ${pattern}
          )`,
        );
      }

      const where = conditions.length > 0 ? and(...conditions) : undefined;

      // Determine sort column
      const sortColumnMap: Record<string, any> = {
        createdAt: auditLogsTable.createdAt,
        action: auditLogsTable.action,
        method: auditLogsTable.method,
        responseStatus: auditLogsTable.responseStatus,
      };
      const sortColumn = sortColumnMap[sortField] ?? auditLogsTable.createdAt;
      const orderFn = sortOrder === "asc" ? sortColumn : desc(sortColumn);

      // Fetch total count
      const [{ count }] = await db
        .select({ count: sql<number>`count(*)::int` })
        .from(auditLogsTable)
        .where(where);

      // Fetch paginated rows with user name join
      const rows = await db
        .select({
          id: auditLogsTable.id,
          userId: auditLogsTable.userId,
          userName: usersTable.name,
          userEmail: usersTable.email,
          action: auditLogsTable.action,
          method: auditLogsTable.method,
          path: auditLogsTable.path,
          routeParams: auditLogsTable.routeParams,
          responseStatus: auditLogsTable.responseStatus,
          ipAddress: auditLogsTable.ipAddress,
          userAgent: auditLogsTable.userAgent,
          requestId: auditLogsTable.requestId,
          createdAt: auditLogsTable.createdAt,
        })
        .from(auditLogsTable)
        .leftJoin(usersTable, eq(auditLogsTable.userId, usersTable.id))
        .where(where)
        .orderBy(orderFn)
        .limit(pageSize)
        .offset(offset);

      return reply.send({
        data: rows,
        pagination: {
          page,
          pageSize,
          total: count,
          totalPages: Math.ceil(count / pageSize),
        },
      });
    },
  });
}

export default auditLogRoutes;
