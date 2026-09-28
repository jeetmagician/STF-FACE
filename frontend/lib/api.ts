import type {
  AnalyzeResponse,
  ApiErrorPayload,
  DatabaseSearchResponse,
  DatabaseSearchStatus,
  DetectResponse,
  HealthResponse,
} from "./types";

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const API_KEY = process.env.NEXT_PUBLIC_API_KEY;

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly payload: ApiErrorPayload;

  constructor(status: number, payload: ApiErrorPayload) {
    super(payload.detail || "Request failed");
    this.name = "ApiError";
    this.status = status;
    this.code = payload.error;
    this.payload = payload;
  }
}

function authHeaders(): HeadersInit {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function parse<T>(response: Response): Promise<T> {
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    throw new ApiError(response.status, {
      error: "invalid_response",
      detail:
        "The server returned a response that could not be read. It may be " +
        "unreachable or misconfigured.",
    });
  }

  if (!response.ok) {
    throw new ApiError(response.status, body as ApiErrorPayload);
  }
  return body as T;
}

function networkError(error: unknown): never {
  if (error instanceof ApiError) throw error;
  throw new ApiError(0, {
    error: "network_error",
    detail:
      `Could not reach the analysis service at ${API_URL}. Check that the ` +
      "backend is running and that NEXT_PUBLIC_API_URL points at it.",
  });
}

export async function getHealth(): Promise<HealthResponse> {
  try {
    const response = await fetch(`${API_URL}/api/health`, {
      headers: authHeaders(),
      cache: "no-store",
    });
    return await parse<HealthResponse>(response);
  } catch (error) {
    return networkError(error);
  }
}

export async function detectFaces(file: File): Promise<DetectResponse> {
  const form = new FormData();
  form.append("image", file);
  try {
    const response = await fetch(`${API_URL}/api/face-detect`, {
      method: "POST",
      body: form,
      headers: authHeaders(),
    });
    return await parse<DetectResponse>(response);
  } catch (error) {
    return networkError(error);
  }
}

export interface AnalyzeArgs {
  oldPhotos: File[];
  newPhotos: File[];
  oldFaceIndices?: number[];
  newFaceIndices?: number[];
  includeVisualisations?: boolean;
}

export async function analyze(args: AnalyzeArgs): Promise<AnalyzeResponse> {
  const form = new FormData();
  args.oldPhotos.forEach((file) => form.append("old_photos", file));
  args.newPhotos.forEach((file) => form.append("new_photos", file));

  if (args.oldFaceIndices) {
    form.append("old_face_indices", JSON.stringify(args.oldFaceIndices));
  }
  if (args.newFaceIndices) {
    form.append("new_face_indices", JSON.stringify(args.newFaceIndices));
  }
  form.append(
    "include_visualisations",
    String(args.includeVisualisations ?? true),
  );

  try {
    const response = await fetch(`${API_URL}/api/analyze`, {
      method: "POST",
      body: form,
      headers: authHeaders(),
    });
    return await parse<AnalyzeResponse>(response);
  } catch (error) {
    return networkError(error);
  }
}

export async function getDatabaseSearchStatus(): Promise<DatabaseSearchStatus> {
  try {
    const response = await fetch(`${API_URL}/api/database-search/status`, {
      headers: authHeaders(),
      cache: "no-store",
    });
    return await parse<DatabaseSearchStatus>(response);
  } catch (error) {
    return networkError(error);
  }
}

export interface DatabaseSearchArgs {
  file: File;
  faceIndex?: number;
  topN?: number;
  refreshIndex?: boolean;
}

export async function searchDatabase(
  args: DatabaseSearchArgs,
): Promise<DatabaseSearchResponse> {
  const form = new FormData();
  form.append("image", args.file);
  if (args.faceIndex !== undefined) {
    form.append("face_index", String(args.faceIndex));
  }
  if (args.topN !== undefined) {
    form.append("top_n", String(args.topN));
  }
  if (args.refreshIndex !== undefined) {
    form.append("refresh_index", String(args.refreshIndex));
  }

  try {
    const response = await fetch(`${API_URL}/api/database-search`, {
      method: "POST",
      body: form,
      headers: authHeaders(),
    });
    return await parse<DatabaseSearchResponse>(response);
  } catch (error) {
    return networkError(error);
  }
}

/** Ask the backend to discard any images it is holding for face selection. */
export async function discardSession(token: string): Promise<void> {
  try {
    await fetch(`${API_URL}/api/session/${encodeURIComponent(token)}`, {
      method: "DELETE",
      headers: authHeaders(),
      keepalive: true,
    });
  } catch {
    // Best effort. The backend expires the entry on its own TTL regardless.
  }
}

export const apiBaseUrl = API_URL;
