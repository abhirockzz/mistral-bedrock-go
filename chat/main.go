package main

import (
	"bufio"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

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

const userMessageFormat = "[INST] %s [/INST]"
const modelID8X7BInstruct = "mistral.mixtral-8x7b-instruct-v0:1"
const bos = "<s>"  //beginning of string - only needed once at the start
const eos = "</s>" // end of a single conversation exchange

var verbose *bool

func main() {
	verbose = flag.Bool("verbose", false, "setting to true will log messages being exchanged with LLM")
	flag.Parse()

	reader := bufio.NewReader(os.Stdin)

	first := true
	var msg string

	for {
		fmt.Print("\nEnter your message: ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if first {
			msg = bos + fmt.Sprintf(userMessageFormat, input)
		} else {
			msg = msg + fmt.Sprintf(userMessageFormat, input)
		}

		payload := MistralRequest{
			Prompt: msg,
		}

		response, err := send(payload)

		if err != nil {
			log.Fatal(err)
		}
		fmt.Println("[Assistant]:", response)

		msg = msg + response + eos + " "

		first = false

	}
}

func send(payload MistralRequest) (string, error) {

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return "", err
	}

	if *verbose {
		fmt.Println("[request payload]", string(payloadBytes))
	}

	output, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(modelID8X7BInstruct),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return "", err
	}

	var resp MistralResponse

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		return "", err
	}

	if *verbose {
		fmt.Println("[response payload]", string(output.Body))
	}

	return resp.Outputs[0].Text, nil
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
