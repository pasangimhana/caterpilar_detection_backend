# Deploy fast api backend to azure vm

Short description about your project, what it does, and what it's used for.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine or server for development and testing purposes.

### Prerequisites

- A virtual machine running Ubuntu (or any other Linux distribution).
- Python 3 installed.
- Nginx installed.
- A private key for SSH access to your server.

### Installing and Running

Follow these steps to get your development environment running:

1. **Copy the Private Key to Your WSL Home Directory**: 
   ```bash
   cp /mnt/e/Assiment\ Hub/butterly\ detection\ reseach/research_key.pem ~/
   ```

2. **Change the Permissions on the Copied Key**: 
   ```bash
   chmod 600 ~/research_key.pem
   ```

3. **Securely Copy Your App to the Server**: 
   ```bash
   scp -i ~/research_key.pem -r /mnt/e/Assiment\ Hub/butterly\ detection\ reseach/app azureuser@52.184.86.31:/home/azureuser/
   ```

4. **Connect to Your Server and Navigate to Your App Directory**: 
   ```bash
   ssh -i ~/research_key.pem azureuser@52.184.86.31
   cd /home/azureuser/your_app_directory
   ```

5. **Create and Activate a Python Virtual Environment**: 
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

6. **Install the Application Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```

7. **Run the FastAPI Application**: 
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

8. **Configure Nginx to Reverse Proxy to Your Application**: 
   - Create a new Nginx server block configuration file.
   - Add the following configuration, adjusting `server_name` and other paths as necessary:
     ```nginx
     server {
         listen 80;
         server_name your_vm_ip;

         location / {
             proxy_pass http://localhost:8000;
             proxy_set_header Host $host;
             proxy_set_header X-Real-IP $remote_addr;
             proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
             proxy_set_header X-Forwarded-Proto $scheme;
         }
     }
     ```

9. **Enable the Nginx Configuration and Restart Nginx**: 
   ```bash
   sudo ln -s /etc/nginx/sites-available/myfastapiapp /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

10. **Create a Systemd Service File for Your Application**: 
    ```bash
    sudo nano /etc/systemd/system/my_application.service
    ```
    - Add the following content, adjusted to your setup:
      ```ini
      [Unit]
      Description=My Application
      After=network.target

      [Service]
      User=azureuser
      WorkingDirectory=/home/azureuser/apps
      ExecStart=/home/azureuser/apps/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
      Restart=always

      [Install]
      WantedBy=multi-user.target
      ```

11. **Enable and Start the Systemd Service**: 
    ```bash
    sudo systemctl daemon-reload
    sudo systemctl enable my_application
    sudo systemctl start my_application
    ```

12. **Check the Status of Your Service**: 
    ```bash
    sudo systemctl status my_application
    ```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Authors

- **thellmike* - *Initial work* - [YourGithubUsername](https://github.com/thellmike)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc
[Watch the Video](https://youtu.be/G2KLLwd7590?si=0_u_kBpD4N9hGZ6q)
