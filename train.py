import torch
import os
import torch.optim as optim
from model import Generator, Discriminator
from loss import adversarial_loss, cycle_consistency_loss
from util import TextDataset, load_data
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

os.makedirs('./output', exist_ok=True)

standard_japanese_sentences = load_data('japanese.json')
kansai_dialect_sentences = load_data('kansai.json')

tokenizer = BertTokenizer.from_pretrained('./bert', local_files_only=True)
vocab_size = tokenizer.vocab_size
dataset = TextDataset(standard_japanese_sentences, kansai_dialect_sentences, tokenizer)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

embedding_layer = BertModel.from_pretrained('./bert', local_files_only=True)
for param in embedding_layer.parameters():
    param.requires_grad = False

G_A_to_B = Generator(vocab_size=vocab_size, embedding_layer=embedding_layer).to(device)
G_B_to_A = Generator(vocab_size=vocab_size, embedding_layer=embedding_layer).to(device)
D_A = Discriminator(vocab_size=vocab_size, embedding_layer=embedding_layer).to(device)
D_B = Discriminator(vocab_size=vocab_size, embedding_layer=embedding_layer).to(device)
optimizer_G = optim.Adam(list(G_A_to_B.parameters()) + list(G_B_to_A.parameters()), lr=0.001, betas=(0.5, 0.999))
optimizer_D_A = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_B = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

num_epochs = 10
iteration = 0
for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        # G: generate fake data
        fake_B = G_A_to_B(real_A)
        fake_A = G_B_to_A(real_B)

        # D: evaluate the fake data
        real_output_A = D_A(real_A)
        fake_output_A = D_A(fake_A.detach())
        real_output_B = D_B(real_B)
        fake_output_B = D_B(fake_B.detach())

        # compute the loss for D
        loss_D_A = adversarial_loss(real_output_A, torch.ones_like(real_output_A)) + adversarial_loss(fake_output_A, torch.zeros_like(fake_output_A))
        loss_D_B = adversarial_loss(real_output_B, torch.ones_like(real_output_B)) + adversarial_loss(fake_output_B, torch.zeros_like(fake_output_B))

        # optimize D
        optimizer_D_A.zero_grad()
        optimizer_D_B.zero_grad()
        loss_D_A.backward()
        loss_D_B.backward()
        optimizer_D_A.step()
        optimizer_D_B.step()

        # G: build cycle
        fake_B = G_A_to_B(real_A)
        cycle_A = G_B_to_A(fake_B)
        fake_A = G_B_to_A(real_B)
        cycle_B = G_A_to_B(fake_A)

        # compute the loss for G
        loss_G_A_to_B = adversarial_loss(D_B(fake_B), torch.ones_like(D_B(fake_B)))
        loss_G_B_to_A = adversarial_loss(D_A(fake_A), torch.ones_like(D_A(fake_A)))
        loss_cycle_A = cycle_consistency_loss(real_A, cycle_A, embedding_layer)
        loss_cycle_B = cycle_consistency_loss(real_B, cycle_B, embedding_layer)

        # total loss for G
        loss_G = loss_G_A_to_B + loss_G_B_to_A + loss_cycle_A + loss_cycle_B

        # optimize G
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        iteration += 1

        if iteration % 100 == 0:
            print(f"Iteration [{iteration}], Epoch [{epoch}/{num_epochs}], Loss D: {loss_D_A.item() + loss_D_B.item()}, Loss G: {loss_G.item()}")

    # save the model
    torch.save(G_A_to_B.state_dict(), f"./output/G_A_to_B_{epoch}.pt")
    torch.save(G_B_to_A.state_dict(), f"./output/G_B_to_A_{epoch}.pt")
    torch.save(D_A.state_dict(), f"./output/D_A_{epoch}.pt")
    torch.save(D_B.state_dict(), f"./output/D_B_{epoch}.pt")

