import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# Email you want to send the update from (only works with gmail)
fromEmail = '***********@gmail.com'
fromEmailPassword = '*************'

# Email you want to send the update to
toEmail = '**********@gmail.com'

def sendEmail(image):
    img_data = open(image, 'rb').read()
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'Security Update'
    msgRoot['From'] = fromEmail
    msgRoot['To'] = toEmail
    msgRoot.preamble = 'Raspberry pi security camera update'

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    msgText = MIMEText('Smart security cam found unknown person')
    msgAlternative.attach(msgText)

    # 	msgText = MIMEText('<img src="cid:image1">', 'html')
    # 	msgAlternative.attach(msgText)

    text = MIMEText("Smart security cam found unknown persons/")
    msgRoot.attach(text)
    msgImage = MIMEImage(img_data)
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)

    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.ehlo()
    smtp.starttls()
    smtp.ehlo()
    smtp.login(fromEmail, fromEmailPassword)
    smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
    smtp.quit()
    print("sent")	
