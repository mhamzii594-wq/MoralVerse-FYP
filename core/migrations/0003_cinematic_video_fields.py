from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_storyrequest_user'),
    ]

    operations = [
        migrations.AddField(
            model_name='storyrequest',
            name='video_type',
            field=models.CharField(
                choices=[('slideshow', 'Slideshow'), ('cinematic', 'Cinematic')],
                default='slideshow',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='storyrequest',
            name='video_progress',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='storyscene',
            name='video_clip_path',
            field=models.CharField(blank=True, max_length=500, null=True),
        ),
    ]
