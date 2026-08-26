import type { PendingCatalogCapture } from './types';

const DATABASE_NAME = 'mjtensu-tile-catalog-capture-v2';
const DATABASE_VERSION = 1;
const STORE_NAME = 'pending-captures';

export async function putPendingCapture(capture: PendingCatalogCapture): Promise<void> {
  const database = await openDatabase();
  try {
    await transactionRequest(database, 'readwrite', (store) => store.put(capture));
  } finally {
    database.close();
  }
}

export async function deletePendingCapture(id: string): Promise<void> {
  const database = await openDatabase();
  try {
    await transactionRequest(database, 'readwrite', (store) => store.delete(id));
  } finally {
    database.close();
  }
}

export async function listPendingCaptures(): Promise<PendingCatalogCapture[]> {
  const database = await openDatabase();
  try {
    return await transactionRequest<PendingCatalogCapture[]>(
      database,
      'readonly',
      (store) => store.getAll() as IDBRequest<PendingCatalogCapture[]>,
    );
  } finally {
    database.close();
  }
}

export async function countPendingCaptures(): Promise<number> {
  const database = await openDatabase();
  try {
    return await transactionRequest<number>(database, 'readonly', (store) => store.count());
  } finally {
    database.close();
  }
}

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE_NAME, DATABASE_VERSION);
    request.onupgradeneeded = () => {
      const database = request.result;
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB open failed.'));
    request.onblocked = () => reject(new Error('IndexedDB upgrade is blocked by another tab.'));
  });
}

function transactionRequest<T>(
  database: IDBDatabase,
  mode: IDBTransactionMode,
  createRequest: (store: IDBObjectStore) => IDBRequest<T>,
): Promise<T> {
  return new Promise((resolve, reject) => {
    const transaction = database.transaction(STORE_NAME, mode);
    const request = createRequest(transaction.objectStore(STORE_NAME));
    let result!: T;
    request.onsuccess = () => {
      result = request.result;
    };
    request.onerror = () => reject(request.error ?? new Error('IndexedDB request failed.'));
    transaction.oncomplete = () => resolve(result);
    transaction.onabort = () => reject(transaction.error ?? new Error('IndexedDB transaction aborted.'));
    transaction.onerror = () => reject(transaction.error ?? new Error('IndexedDB transaction failed.'));
  });
}
