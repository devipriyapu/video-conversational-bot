export interface UploadRequest {
  youtube_url: string;
  collection_name?: string;
}

export interface UploadResponse {
  video_id: string;
  title: string;
  collection_name: string;
  chunk_count: number;
  transcript_path: string;
}

export interface ChatRequest {
  question: string;
  video_id?: string;
  collection_name?: string;
  top_k?: number;
}

export interface SourceChunk {
  text: string;
  metadata: Record<string, unknown>;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: SourceChunk[];
  tokens_used: number;
}

export interface ChatMessage {
  sender: 'user' | 'assistant';
  text: string;
  sources?: SourceChunk[];
}
