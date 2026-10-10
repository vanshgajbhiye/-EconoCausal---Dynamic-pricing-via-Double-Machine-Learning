import mongoose from 'mongoose';

export let isConnectedToMongo = false;

export const connectDB = async () => {
  try {
    mongoose.set('bufferCommands', false);
    const conn = await mongoose.connect(process.env.MONGO_URI || 'mongodb://127.0.0.1:27017/econocausal', {
      serverSelectionTimeoutMS: 2000,
    });
    isConnectedToMongo = true;
    console.log(`[Database] MongoDB Connected: ${conn.connection.host}`);
  } catch (error) {
    isConnectedToMongo = false;
    console.warn(`[Database Warning] Could not connect to MongoDB: ${error.message}`);
    console.warn('[Database] Running in in-memory dev mode with seed products & simulated DML.');
  }
};

