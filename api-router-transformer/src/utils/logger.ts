import winston from 'winston';
import { config } from '../config';

export const logger = winston.createLogger({
  level: config.logLevel,
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.printf(({ timestamp, level, message, requestId }) => {
      const id = requestId ? `[${requestId}]` : '';
      return `${timestamp} [${level}] ${id}: ${message}`;
    })
  ),
  transports: [new winston.transports.Console()],
});

export function createRequestLogger(requestId: string) {
  return logger.child({ requestId });
}
