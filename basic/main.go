package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

const defaultRegion = "us-east-1"

var brc *bedrockruntime.Client

func init() {

	region := os.Getenv("AWS_REGION")
	if region == "" {
		region = defaultRegion
	}

	cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
	if err != nil {
		log.Fatal(err)
	}

	brc = bedrockruntime.NewFromConfig(cfg)
}

const modelID7BInstruct = "mistral.mistral-7b-instruct-v0:2"

const promptFormat = "<s>[INST] %s [/INST]"

func main() {

	msg := "Hello, what's your name?"

	payload := MistralRequest{
		Prompt: fmt.Sprintf(promptFormat, msg),
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("request payload:\n", string(payloadBytes))

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(modelID7BInstruct),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		log.Fatal(err)
	}

	var resp MistralResponse

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("response payload:\n", string(output.Body))

	fmt.Println("response string:\n", resp.Outputs[0].Text)

}

type MistralRequest struct {
	Prompt        string   `json:"prompt"`
	MaxTokens     int      `json:"max_tokens,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	TopP          float64  `json:"top_p,omitempty"`
	TopK          int      `json:"top_k,omitempty"`
	StopSequences []string `json:"stop,omitempty"`
}

type MistralResponse struct {
	Outputs []Outputs `json:"outputs"`
}
type Outputs struct {
	Text       string `json:"text"`
	StopReason string `json:"stop_reason"`
}
