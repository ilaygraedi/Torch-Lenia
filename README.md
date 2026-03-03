#Torch-Lenia: GPU-Accelerated Artificial Life 
מנוע סימולציה של אוטומטים תאיים רציפים (Continuous Cellular Automata) שמדמה התפתחות של יצורים וירטואליים. 
הפרויקט נכתב במקור ב-NumPy, אך שוכתב לחלוטין ל-PyTorch כדי לרוץ על כרטיס המסך (GPU) ולאפשר חישובים של מיליוני פיקסלים בזמן אמת

![20260303-1621-54 1118620](https://github.com/user-attachments/assets/f4a47e67-a0eb-4afe-837c-1863a19d871a)

*Multi-Channel Interaction: תמיכה ב-3 ערוצי צבע (RGB) עם אינטראקציה צולבת (Cross-channel) באמצעות כפל מטריצות.

*GPU Acceleration: שימוש ב-PyTorch ו-CUDA כדי להעביר את עומס החישוב מה-CPU לכרטיס המסך, מה שמאפשר הרצה של 60 פריימים בשנייה על עולם של 800x800.

*Fast Fourier Transform (FFT): שימוש ב- torch.fft כדי לחשב את הקונבולוציות וההשפעה הסביבתית של כל תא בצורה יעילה במקום לולאות איטיות.

איזה ספריות צריך להתקין: pip install torch torchvision pygame numpy

הפקודה להרצה: python main.py

הסבר קצר על הממשק: "קליק שמאלי של העכבר מצייר 'חומר ביולוגי' על המסך, ונותן לסימולציה לרוץ".
