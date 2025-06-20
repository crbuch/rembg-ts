/**
 * Session registry and management
 */

import { BaseSession } from '../sessions/base';
import { U2netSession } from '../sessions/u2net';
import { U2netpSession } from '../sessions/u2netp';

export const sessions: Record<string, typeof BaseSession> = {};

// Register session classes
sessions[U2netSession.getModelName()] = U2netSession;
sessions[U2netpSession.getModelName()] = U2netpSession;

export const sessions_names = Object.keys(sessions);
export const sessions_class = Object.values(sessions);
