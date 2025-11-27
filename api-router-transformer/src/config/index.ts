import dotenv from 'dotenv';
dotenv.config();

export interface Config {
  port: number;
  backendUrl: string;
  backendApiKey: string;
  logLevel: string;
  corsEnabled: boolean;
  requestTimeout: number;
}

export const config: Config = {
  port: parseInt(process.env.PORT || '8000', 10),
  backendUrl: process.env.BACKEND_URL || '',
  backendApiKey: process.env.BACKEND_API_KEY || '',
  logLevel: process.env.LOG_LEVEL || 'info',
  corsEnabled: (process.env.CORS_ENABLED || 'true') === 'true',
  requestTimeout: parseInt(process.env.REQUEST_TIMEOUT || '60000', 10),
};

export function validateConfig() {
  const errors: string[] = [];
  if (!config.backendUrl) errors.push('BACKEND_URL is required');
  if (!config.backendApiKey) errors.push('BACKEND_API_KEY is required');
  return { valid: errors.length === 0, errors };
}
