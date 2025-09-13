from django.db import models

class Document(models.Model):
    file = models.FileField(upload_to='documents/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    original_name = models.CharField(max_length=255, blank=True, null=True)

    def __str__(self):
        return f"Document {self.id} - {self.original_name or self.file.name}"
