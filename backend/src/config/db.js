/**
 * backend/src/config/db.js
 * ─────────────────────────
 * Re-exports the shared DB connection from the database/ layer.
 * The actual logic lives at: database/config/connection.js
 */
export { connectDB, disconnectDB, isConnectedToMongo } from '../../../database/config/connection.js';
