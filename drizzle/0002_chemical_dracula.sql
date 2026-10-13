CREATE TABLE `cq_password_attempts` (
	`attempt_key` text PRIMARY KEY NOT NULL,
	`attempts` integer NOT NULL,
	`expires_at` integer NOT NULL
);
--> statement-breakpoint
CREATE INDEX `cq_password_attempts_expiry_idx` ON `cq_password_attempts` (`expires_at`);--> statement-breakpoint
CREATE TABLE `cq_password_sessions` (
	`token_hash` text PRIMARY KEY NOT NULL,
	`user_id` text NOT NULL,
	`created_at` integer NOT NULL,
	`expires_at` integer NOT NULL,
	FOREIGN KEY (`user_id`) REFERENCES `cq_password_users`(`id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `cq_password_sessions_expiry_idx` ON `cq_password_sessions` (`expires_at`);--> statement-breakpoint
CREATE INDEX `cq_password_sessions_user_idx` ON `cq_password_sessions` (`user_id`,`created_at`);--> statement-breakpoint
CREATE TABLE `cq_password_users` (
	`id` text PRIMARY KEY NOT NULL,
	`username` text NOT NULL,
	`username_key` text NOT NULL,
	`password_salt` text NOT NULL,
	`password_hash` text NOT NULL,
	`password_scheme` text NOT NULL,
	`created_at` integer NOT NULL
);
--> statement-breakpoint
CREATE UNIQUE INDEX `cq_password_users_username_key_unique` ON `cq_password_users` (`username_key`);