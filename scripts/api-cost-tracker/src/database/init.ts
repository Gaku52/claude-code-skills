#!/usr/bin/env node

import { db } from './index';

console.log('[Init] Initializing database...');
db.init();
console.log('[Init] Database initialized successfully');
