import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../environments/environment';
import { ChatRequest, ChatResponse, UploadRequest, UploadResponse } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = environment.apiBaseUrl;

  constructor(private readonly http: HttpClient) {}

  uploadVideo(payload: UploadRequest): Observable<UploadResponse> {
    return this.http.post<UploadResponse>(`${this.baseUrl}/upload`, payload);
  }

  askQuestion(payload: ChatRequest): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, payload);
  }

  deleteCollection(videoId?: string, collectionName?: string): Observable<{ message: string }> {
    return this.http.request<{ message: string }>('DELETE', `${this.baseUrl}/upload/collection`, {
      body: {
        video_id: videoId,
        collection_name: collectionName,
      },
    });
  }
}
