import { index, integer, primaryKey, sqliteTable, text } from "drizzle-orm/sqlite-core";

export const users = sqliteTable("users", {
  id: text("id").primaryKey(),
  username: text("username").notNull().unique(),
  displayName: text("display_name").notNull(),
  passwordSalt: text("password_salt").notNull(),
  passwordHash: text("password_hash").notNull(),
  passwordIterations: integer("password_iterations").notNull().default(100000),
  createdAt: integer("created_at").notNull(),
});

export const sessions = sqliteTable("sessions", {
  tokenHash: text("token_hash").primaryKey(),
  userId: text("user_id").notNull().references(() => users.id, { onDelete: "cascade" }),
  createdAt: integer("created_at").notNull(),
  expiresAt: integer("expires_at").notNull(),
}, (table) => [index("sessions_expiry_idx").on(table.expiresAt)]);

export const authAttempts = sqliteTable("auth_attempts", {
  attemptKey: text("attempt_key").primaryKey(),
  failures: integer("failures").notNull(),
  expiresAt: integer("expires_at").notNull(),
}, (table) => [index("auth_attempts_expiry_idx").on(table.expiresAt)]);

export const userProgress = sqliteTable("user_progress", {
  userId: text("user_id").primaryKey().references(() => users.id, { onDelete: "cascade" }),
  payload: text("payload").notNull(),
  revision: integer("revision").notNull().default(1),
  updatedAt: integer("updated_at").notNull(),
});

// Original local-password tables remain untouched for migration compatibility.
// New records are scoped to the Site dispatch identity, never a client user id.
export const cloudStates = sqliteTable("cq_cloud_states", {
  userId: text("user_id").primaryKey(),
  payload: text("payload"),
  revision: integer("revision").notNull().default(0),
  updatedAt: text("updated_at"),
});

export const cloudMutations = sqliteTable("cq_cloud_mutations", {
  userId: text("user_id").notNull().references(() => cloudStates.userId, { onDelete: "cascade" }),
  mutationId: text("mutation_id").notNull(),
  baseRevision: integer("base_revision").notNull(),
  payloadHash: text("payload_hash").notNull(),
  revision: integer("revision").notNull(),
  createdAt: text("created_at").notNull(),
}, (table) => [
  primaryKey({ columns: [table.userId, table.mutationId] }),
  index("cq_cloud_mutations_revision_idx").on(table.userId, table.revision),
]);
