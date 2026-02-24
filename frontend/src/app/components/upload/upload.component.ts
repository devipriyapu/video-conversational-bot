import { Component, EventEmitter, Output } from '@angular/core';
import { FormControl, FormGroup, ReactiveFormsModule, Validators } from '@angular/forms';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
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
    MatProgressSpinnerModule,
  ],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css',
})
export class UploadComponent {
  @Output() indexed = new EventEmitter<UploadResponse>();

  readonly form = new FormGroup({
    youtube_url: new FormControl('', [Validators.required]),
    collection_name: new FormControl(''),
  });

  loading = false;
  error = '';
  success = '';

  constructor(private readonly apiService: ApiService) {}

  submit(): void {
    if (this.form.invalid || this.loading) {
      return;
    }

    this.error = '';
    this.success = '';
    this.loading = true;

    const youtubeUrl = this.form.controls.youtube_url.value ?? '';
    const collectionName = this.form.controls.collection_name.value ?? '';

    this.apiService
      .uploadVideo({ youtube_url: youtubeUrl, collection_name: collectionName || undefined })
      .pipe(finalize(() => (this.loading = false)))
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
}
