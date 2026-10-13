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

// Historical tables and ChatGPT archives remain untouched. Cloud records for
// native password accounts use a separate cqp_ user-id namespace.
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

// A fresh schema avoids treating any legacy password material as a new account.
export const passwordUsers = sqliteTable("cq_password_users", {
  id: text("id").primaryKey(),
  username: text("username").notNull(),
  usernameKey: text("username_key").notNull().unique(),
  passwordSalt: text("password_salt").notNull(),
  passwordHash: text("password_hash").notNull(),
  passwordScheme: text("password_scheme").notNull(),
  createdAt: integer("created_at").notNull(),
});

export const passwordSessions = sqliteTable("cq_password_sessions", {
  tokenHash: text("token_hash").primaryKey(),
  userId: text("user_id").notNull().references(() => passwordUsers.id, { onDelete: "cascade" }),
  createdAt: integer("created_at").notNull(),
  expiresAt: integer("expires_at").notNull(),
}, (table) => [
  index("cq_password_sessions_expiry_idx").on(table.expiresAt),
  index("cq_password_sessions_user_idx").on(table.userId, table.createdAt),
]);

export const passwordAttempts = sqliteTable("cq_password_attempts", {
  attemptKey: text("attempt_key").primaryKey(),
  attempts: integer("attempts").notNull(),
  expiresAt: integer("expires_at").notNull(),
}, (table) => [index("cq_password_attempts_expiry_idx").on(table.expiresAt)]);
