import { Component, EventEmitter, OnDestroy, Output } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { Subscription } from 'rxjs';
import { finalize } from 'rxjs';

import { ApiService } from '../../services/api.service';
import { UploadResponse } from '../../models/api.models';

@Component({
  selector: 'app-upload',
  standalone: true,
  imports: [
    ReactiveFormsModule,
    MatButtonModule,
    MatCardModule,
    MatFormFieldModule,
    MatInputModule,
    MatProgressBarModule,
  ],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css',
})
export class UploadComponent implements OnDestroy {
  @Output() indexed = new EventEmitter<UploadResponse>();
  @Output() uploadingChange = new EventEmitter<boolean>();

  readonly form = new FormGroup({
    youtube_url: new FormControl('https://www.youtube.com/watch?v=uMzUB89uSxU', [Validators.required]),
    collection_name: new FormControl(''),
  });

  loading = false;
  progress = 0;
  error = '';
  success = '';
  embedUrl: SafeResourceUrl | null = null;
  private progressIntervalId: ReturnType<typeof setInterval> | null = null;
  private youtubeUrlSub: Subscription;

  constructor(
    private readonly apiService: ApiService,
    private readonly sanitizer: DomSanitizer,
  ) {
    this.updateEmbedUrl(this.form.controls.youtube_url.value ?? '');
    this.youtubeUrlSub = this.form.controls.youtube_url.valueChanges.subscribe((url) => {
      this.updateEmbedUrl(url ?? '');
    });
  }

  submit(): void {
    if (this.form.invalid || this.loading) {
      return;
    }

    this.error = '';
    this.success = '';
    this.loading = true;
    this.progress = 4;
    this.uploadingChange.emit(true);
    this.startFakeProgress();

    const youtubeUrl = this.form.controls.youtube_url.value ?? '';
    const collectionName = this.form.controls.collection_name.value ?? '';

    this.apiService
      .uploadVideo({ youtube_url: youtubeUrl, collection_name: collectionName || undefined })
      .pipe(
        finalize(() => {
          this.stopFakeProgress();
          this.loading = false;
          this.progress = 100;
          this.uploadingChange.emit(false);
        }),
      )
      .subscribe({
        next: (res) => {
          this.success = `Indexed ${res.chunk_count} chunks for video ${res.video_id}`;
          this.indexed.emit(res);
        },
        error: (err) => {
          const detail = err?.error?.detail ?? err?.message ?? 'Upload failed';
          const status = err?.status ? `HTTP ${err.status}` : 'HTTP ?';
          this.error = `${status}: ${detail}`;
        },
      });
  }

  ngOnDestroy(): void {
    this.stopFakeProgress();
    this.youtubeUrlSub.unsubscribe();
  }

  private startFakeProgress(): void {
    this.stopFakeProgress();
    this.progressIntervalId = setInterval(() => {
      if (!this.loading) {
        return;
      }

      this.progress = Math.min(this.progress + 5, 95);
    }, 500);
  }

  private stopFakeProgress(): void {
    if (this.progressIntervalId) {
      clearInterval(this.progressIntervalId);
      this.progressIntervalId = null;
    }
  }

  private extractVideoId(url: string): string {
    try {
      const parsed = new URL(url.trim());
      const host = parsed.hostname.replace(/^www\./, '');

      if (host === 'youtu.be') {
        return parsed.pathname.replace('/', '').split('/')[0];
      }

      if (host === 'youtube.com' || host === 'm.youtube.com') {
        if (parsed.pathname === '/watch') {
          return parsed.searchParams.get('v') ?? '';
        }

        if (parsed.pathname.startsWith('/embed/')) {
          return parsed.pathname.split('/')[2] ?? '';
        }
      }
    } catch {
      return '';
    }

    return '';
  }

  private updateEmbedUrl(rawUrl: string): void {
    const videoId = this.extractVideoId(rawUrl);
    if (!videoId) {
      this.embedUrl = null;
      return;
    }

    this.embedUrl = this.sanitizer.bypassSecurityTrustResourceUrl(`https://www.youtube.com/embed/${videoId}`);
  }
}
