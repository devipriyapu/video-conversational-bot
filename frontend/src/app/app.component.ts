import { Component } from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { UploadResponse } from './models/api.models';
import { ApiService } from './services/api.service';
import { ChatComponent } from './components/chat/chat.component';
import { UploadComponent } from './components/upload/upload.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [UploadComponent, ChatComponent, MatButtonModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  indexedVideoId = '';
  collectionName = '';
  deleteMessage = '';
  deleteError = '';

  constructor(private readonly apiService: ApiService) {}

  onIndexed(event: UploadResponse): void {
    this.indexedVideoId = event.video_id;
    this.collectionName = event.collection_name;
    this.deleteMessage = '';
    this.deleteError = '';
  }

  deleteCollection(): void {
    this.deleteMessage = '';
    this.deleteError = '';

    this.apiService.deleteCollection(this.indexedVideoId, this.collectionName).subscribe({
      next: (res) => {
        this.deleteMessage = res.message;
        this.indexedVideoId = '';
        this.collectionName = '';
      },
      error: (err) => {
        this.deleteError = err?.error?.detail ?? 'Failed to delete collection';
      },
    });
  }
}
