/**
 * backend/src/config/db.js
 * ─────────────────────────
 * MongoDB connection for the backend server.
 * Schema definitions live in database/schemas/
 * Seed scripts live in database/seeds/
 */
import mongoose from 'mongoose';

export let isConnectedToMongo = false;

export const connectDB = async () => {
  try {
    mongoose.set('bufferCommands', false);
    const uri = process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/econocausal';
    const conn = await mongoose.connect(uri, { serverSelectionTimeoutMS: 3000 });
    isConnectedToMongo = true;
    console.log(`[DB] MongoDB connected → ${conn.connection.host}/${conn.connection.name}`);
  } catch (err) {
    isConnectedToMongo = false;
    console.warn(`[DB] MongoDB unavailable: ${err.message}`);
    console.warn('[DB] Running in in-memory fallback mode.');
  }
};

export const disconnectDB = async () => {
  await mongoose.disconnect();
  isConnectedToMongo = false;
};
