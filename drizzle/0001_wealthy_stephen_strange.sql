CREATE TABLE `cq_cloud_mutations` (
	`user_id` text NOT NULL,
	`mutation_id` text NOT NULL,
	`base_revision` integer NOT NULL,
	`payload_hash` text NOT NULL,
	`revision` integer NOT NULL,
	`created_at` text NOT NULL,
	PRIMARY KEY(`user_id`, `mutation_id`),
	FOREIGN KEY (`user_id`) REFERENCES `cq_cloud_states`(`user_id`) ON UPDATE no action ON DELETE cascade
);
--> statement-breakpoint
CREATE INDEX `cq_cloud_mutations_revision_idx` ON `cq_cloud_mutations` (`user_id`,`revision`);--> statement-breakpoint
CREATE TABLE `cq_cloud_states` (
	`user_id` text PRIMARY KEY NOT NULL,
	`payload` text,
	`revision` integer DEFAULT 0 NOT NULL,
	`updated_at` text
);
